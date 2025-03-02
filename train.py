import os
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from my_env.deua_env import DynamicEUAEnv
from nets.attention_net import AttentionNet
from util.torch_utils import seed_torch, create_graph_from_server_list
from util.utils import get_logger
from data.eua_dataset import get_dataset


def train(config):
    train_config, data_config, model_config, env_config = (config['train'], config['data'],
                                                           config['model'], config['env'])
    device = train_config['device'] if torch.cuda.is_available() else 'cpu'
    print('Using device: {}'.format(device))
    dataset = get_dataset(data_config['x_end'], data_config['y_end'], data_config['miu'], data_config['sigma'],
                          data_config['total_sec'], data_config['user_num_per_sec'],
                          data_config['user_stay_miu'], data_config['user_stay_sigma'],
                          data_config['data_size'], data_config['max_cov'], device, train_config['dir_name'])
    train_loader = DataLoader(dataset=dataset['train'], batch_size=train_config['batch_size'], shuffle=False)
    valid_loader = DataLoader(dataset=dataset['valid'], batch_size=train_config['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset=dataset['test'], batch_size=train_config['batch_size'], shuffle=False)

    seed_torch()
    model = AttentionNet(6, 6, hidden_dim=model_config['hidden_dim'], device=device,
                         exploration_c=model_config['exploration_c'],
                         server_embedding_type=model_config['server_embedding_type'],
                         total_sec=data_config['total_sec'])
    # The user stay duration is consistent within each batch during training, allowing batch processing and avoiding loops within a batch, which significantly improves efficiency.
    env = DynamicEUAEnv(data_config['user_num_per_sec'], data_config['total_sec'], env_config['capacity_reward_rate'],
                        user_stay_batch_same=True)
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    train_type = train_config['train_type']

    # Load the model for continued training or fine-tuning.
    if model_config['need_continue']:
        checkpoint = torch.load(model_config['continue_model_filename'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if model_config['continue_lr'] != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = model_config['continue_lr']
        start_epoch = checkpoint['epoch'] + 1

        print("Successfully imported pre-trained model")
    else:
        start_epoch = 0

    # Multiply by lr_decay each epoch.
    lr_scheduler = ExponentialLR(optimizer, train_config['lr_decay'], last_epoch=start_epoch - 1)
    print("Current learning rate:", lr_scheduler.get_last_lr())

    critic_exp_mvg_avg = torch.zeros(1, device=device)

    dir_name = ("" + time.strftime('%m%d%H%M', time.localtime(time.time()))
                + "_server_" + str(data_config['x_end']) + "_" + str(data_config['y_end'])
                + "_miu_" + str(data_config['miu']) + "_sigma_" + str(data_config['sigma'])
                + "_user_num_per_sec_" + str(data_config['user_num_per_sec'])
                + "_user_stay_miu_" + str(data_config['user_stay_miu'])
                + "_" + train_type + "_capa_rate_" + str(env_config['capacity_reward_rate'])
                + "_" + model_config['server_embedding_type'])
    dir_name = os.path.join(train_config['dir_name'], dir_name)
    log_file_name = dir_name + '/log.log'

    os.makedirs(dir_name, exist_ok=True)
    tensorboard_writer = SummaryWriter(dir_name)
    logger = get_logger(log_file_name)
    now_exit = False

    start_time = time.time()
    all_valid_reward_list = []
    all_valid_user_list = []
    all_valid_server_list = []
    all_valid_capacity_list = []
    all_valid_experience_list = []
    best_r = 0
    best_epoch_id = 0
    total_batch_num = 0

    if model_config['server_embedding_type'] == 'gnn':
        graph = create_graph_from_server_list(dataset['train'].servers_tensor, device)
    else:
        graph = None

    for epoch in range(start_epoch, train_config['epochs']):
        # Train
        model.train()
        model.policy = 'sample'
        model.beam_num = 1
        for batch_idx, (server_seq, user_seq, masks) in enumerate(train_loader):
            server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

            reward, actions_probs, _, indicators = model(env, user_seq, server_seq, masks, graph)
            user_allocated_props, server_costs, capacity_used_props, user_experience = indicators[:4]

            tensorboard_writer.add_scalar('train/train_batch_reward', -torch.mean(reward), total_batch_num)
            total_batch_num += 1

            if train_type == 'REINFORCE':
                if batch_idx == 0:
                    critic_exp_mvg_avg = reward.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * train_config['beta']) \
                                         + ((1. - train_config['beta']) * reward.mean())
                advantage = reward - critic_exp_mvg_avg.detach()

            elif train_type == 'SCST':
                model.policy = 'greedy'
                with torch.no_grad():
                    reward2, _, _, _ = model(env, user_seq, server_seq, masks, graph)
                    advantage = reward - reward2
                model.policy = 'sample'

            else:
                raise NotImplementedError

            log_probs = torch.zeros(user_seq.size(0), device=device)
            for prob in actions_probs:
                log_prob = torch.log(prob)
                log_probs += log_prob
            log_probs[log_probs < -1000] = -1000.

            reinforce = torch.dot(advantage.detach(), log_probs)
            actor_loss = reinforce.mean()

            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()

            logger.info(
                'Epoch {}: Train [{}/{} ({:.1f}%)]\tR: {:.6f}\tuser_props: {:.6f}'
                '\tserver_costs: {:.6f}\tcapacity_props: {:.6f}\tuser_experience: {:.6f}'.format(
                    epoch,
                    min((batch_idx + 1) * train_config['batch_size'], data_config['data_size']['train']),
                    data_config['data_size']['train'],
                    100. * (batch_idx + 1) / len(train_loader),
                    -torch.mean(reward),
                    torch.mean(user_allocated_props),
                    torch.mean(server_costs),
                    torch.mean(capacity_used_props),
                    torch.mean(user_experience)
                ))

        tensorboard_writer.add_scalar('train/train_reward', -torch.mean(reward), epoch)
        tensorboard_writer.add_scalar('train/train_user_allocated_props', torch.mean(user_allocated_props), epoch)
        tensorboard_writer.add_scalar('train/train_server_costs', torch.mean(server_costs), epoch)
        tensorboard_writer.add_scalar('train/train_capacity_used_props', torch.mean(capacity_used_props), epoch)
        tensorboard_writer.add_scalar('train/train_user_experience', torch.mean(user_experience), epoch)

        # Valid and Test
        model.eval()
        model.policy = 'greedy'
        logger.info('')
        with (torch.no_grad()):
            # Validation
            valid_R_list = []
            valid_user_allocated_props_list = []
            valid_server_costs_list = []
            valid_capacity_used_props_list = []
            valid_user_experience_list = []
            model.policy = 'greedy'
            model.beam_num = 1
            for batch_idx, (server_seq, user_seq, masks) in enumerate(valid_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                reward, _, _, indicators = model(env, user_seq, server_seq, masks, graph)
                user_allocated_props, server_costs, capacity_used_props, user_experience = indicators[:4]

                logger.info(
                    'Epoch {}: Valid [{}/{} ({:.1f}%)]\tR: {:.6f}\tuser_props: {:.6f}'
                    '\tserver_costs: {:.6f}\tcapacity_props: {:.6f}\tuser_experience: {:.6f}'.format(
                        epoch,
                        min((batch_idx + 1) * train_config['batch_size'], data_config['data_size']['valid']),
                        data_config['data_size']['valid'],
                        100. * (batch_idx + 1) / len(valid_loader),
                        -torch.mean(reward),
                        torch.mean(user_allocated_props),
                        torch.mean(server_costs),
                        torch.mean(capacity_used_props),
                        torch.mean(user_experience)
                    ))

                valid_R_list.append(reward)
                valid_user_allocated_props_list.append(user_allocated_props)
                valid_server_costs_list.append(server_costs)
                valid_capacity_used_props_list.append(capacity_used_props)
                valid_user_experience_list.append(user_experience)

            valid_R_list = torch.cat(valid_R_list)
            valid_user_allocated_props_list = torch.cat(valid_user_allocated_props_list)
            valid_server_costs_list = torch.cat(valid_server_costs_list)
            valid_capacity_used_props_list = torch.cat(valid_capacity_used_props_list)
            valid_user_experience_list = torch.cat(valid_user_experience_list)
            valid_r = torch.mean(valid_R_list)
            valid_user_alloc = torch.mean(valid_user_allocated_props_list)
            valid_server_cos = torch.mean(valid_server_costs_list)
            valid_capacity_use = torch.mean(valid_capacity_used_props_list)
            valid_user_exper = torch.mean(valid_user_experience_list)
            logger.info('Epoch {}: Valid \tR: {:.6f}\tuser_props: {:.6f}\tserver_costs: {:.6f}'
                        '\tcapacity_props: {:.6f}\tuser_experience: {:.6f}'
                        .format(epoch, -valid_r, valid_user_alloc, valid_server_cos, valid_capacity_use,
                                valid_user_exper))

            tensorboard_writer.add_scalar('valid/valid_reward', -valid_r, epoch)
            tensorboard_writer.add_scalar('valid/valid_user_allocated_props', valid_user_alloc, epoch)
            tensorboard_writer.add_scalar('valid/valid_server_costs', valid_server_cos, epoch)
            tensorboard_writer.add_scalar('valid/valid_capacity_used_props', valid_capacity_use, epoch)
            tensorboard_writer.add_scalar('valid/valid_user_experience', valid_user_exper, epoch)

            all_valid_reward_list.append(valid_r)
            all_valid_user_list.append(valid_user_alloc)
            all_valid_server_list.append(valid_server_cos)
            all_valid_capacity_list.append(valid_capacity_use)
            all_valid_experience_list.append(valid_user_exper)

            # Save Model
            model_filename = dir_name + "/" + time.strftime(
                '%m%d%H%M', time.localtime(time.time())
            ) + "_{:.2f}_{:.2f}_{:.2f}_{:.2f}_Epoch_{}".format(all_valid_user_list[best_epoch_id - start_epoch] * 100,
                                                               all_valid_server_list[best_epoch_id - start_epoch] * 100,
                                                               all_valid_capacity_list[
                                                                   best_epoch_id - start_epoch] * 100,
                                                               all_valid_experience_list[best_epoch_id - start_epoch],
                                                               epoch) + '.mdl'
            torch.save(model.state_dict(), model_filename)
            logger.info("Model saved in: {}".format(model_filename))

            if valid_r < best_r:
                best_r = valid_r
                best_epoch_id = epoch
                best_time = 0
                logger.info("best reward in this training session\n")
            else:
                best_time += 1
                logger.info("The performance has not improved for {} epochs.\n".format(best_time))

            # Test
            test_R_list = []
            test_user_allocated_props_list = []
            test_server_costs_list = []
            test_capacity_used_props_list = []
            test_user_experience_list = []
            for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                # Ensure consistency with other methods during testing, meaning the user stay duration varies.
                env.user_stay_batch_same = False
                reward, _, _, indicators = model(env, user_seq, server_seq, masks, graph)
                env.user_stay_batch_same = True
                user_allocated_props, server_costs, capacity_used_props, user_experience = indicators[:4]

                logger.info(
                    'Epoch {}: Test [{}/{} ({:.1f}%)]\tR: {:.6f}\tuser_props: {:.6f}'
                    '\tserver_costs: {:.6f}\tcapacity_props: {:.6f}\tuser_experience: {:.6f}'.format(
                        epoch,
                        min((batch_idx + 1) * train_config['batch_size'], data_config['data_size']['test']),
                        data_config['data_size']['test'],
                        100. * (batch_idx + 1) / len(test_loader),
                        -torch.mean(reward),
                        torch.mean(user_allocated_props),
                        torch.mean(server_costs),
                        torch.mean(capacity_used_props),
                        torch.mean(user_experience)
                    ))

                test_R_list.append(reward)
                test_user_allocated_props_list.append(user_allocated_props)
                test_server_costs_list.append(server_costs)
                test_capacity_used_props_list.append(capacity_used_props)
                test_user_experience_list.append(user_experience)

            test_R_list = torch.cat(test_R_list)
            test_user_allocated_props_list = torch.cat(test_user_allocated_props_list)
            test_server_costs_list = torch.cat(test_server_costs_list)
            test_capacity_used_props_list = torch.cat(test_capacity_used_props_list)
            test_user_experience_list = torch.cat(test_user_experience_list)

            test_r = torch.mean(test_R_list)
            test_user_alloc = torch.mean(test_user_allocated_props_list)
            test_server_cos = torch.mean(test_server_costs_list)
            test_capacity_use = torch.mean(test_capacity_used_props_list)
            test_user_exper = torch.mean(test_user_experience_list)

            logger.info('Epoch {}: Test \tR: {:.6f}\tuser_props: {:.6f}\tserver_costs: {:.6f}'
                        '\tcapacity_props: {:.6f}\tuser_experience: {:.6f}'
                        .format(epoch, -test_r, test_user_alloc, test_server_cos, test_capacity_use, test_user_exper))
            tensorboard_writer.add_scalar('test/test_reward', -test_r, epoch)
            tensorboard_writer.add_scalar('test/test_user_allocated_props', test_user_alloc, epoch)
            tensorboard_writer.add_scalar('test/test_server_costs', test_server_cos, epoch)
            tensorboard_writer.add_scalar('test/test_capacity_used_props', test_capacity_use, epoch)
            tensorboard_writer.add_scalar('test/test_user_experience', test_user_exper, epoch)

        logger.info('')

        # If the validation reward does not improve for the specified number of epochs, stop training.
        if best_time >= train_config['wait_best_reward_epoch']:
            # Save a checkpoint of the model for continued training before exiting
            now_exit = True

        # Learning rate decay.
        lr_scheduler.step()
        logger.info("The learning rate is adjusted to:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        # Save a checkpoint for continued training every interval epochs or before exiting.
        if epoch % train_config['save_model_epoch_interval'] == train_config['save_model_epoch_interval'] - 1 \
                or now_exit:
            model_filename = dir_name + "/" + time.strftime(
                '%m%d%H%M', time.localtime(time.time())
            ) + "_{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(valid_user_alloc * 100,
                                                      valid_server_cos * 100,
                                                      valid_capacity_use * 100,
                                                      valid_user_exper) + '.pt'

            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_filename)
            logger.info("The model has been saved to: {}".format(model_filename))

            if now_exit:
                break

    logger.info("The effect is as follows:")
    for i in range(len(all_valid_reward_list)):
        logger.info("Epoch: {}\treward: {:.6f}\tuser_props: {:.6f}"
                    "\tserver_costs: {:.6f}\tcapacity_props: {:.6f}\tuser_experience: {:.6f}"
                    .format(i + start_epoch, -all_valid_reward_list[i], all_valid_user_list[i],
                            all_valid_server_list[i], all_valid_capacity_list[i], all_valid_experience_list[i]))
    logger.info("Training ended, the {}th epoch had the best effect, the best reward: {:.6f}, user allocation rate: {:.6f},"
                "server cost: {:.6f}, resource utilization: {:.6f}, user experience: {:.6f}"
                .format(best_epoch_id, -best_r,
                        all_valid_user_list[best_epoch_id - start_epoch],
                        all_valid_server_list[best_epoch_id - start_epoch],
                        all_valid_capacity_list[best_epoch_id - start_epoch],
                        all_valid_experience_list[best_epoch_id - start_epoch]))
    end_time = time.time()
    logger.info("Training Time: {:.2f}h".format(((end_time - start_time) / 3600)))
    logger.info("The model has been saved to: {}".format(model_filename))


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        loaded_config = yaml.safe_load(f)
    train(loaded_config)
