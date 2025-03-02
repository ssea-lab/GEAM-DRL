import os
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from my_env.deua_env import DynamicEUAEnv
from nets.attention_net import AttentionNet
from util.torch_utils import seed_torch, create_graph_from_server_list
from util.utils import get_logger
from data.eua_dataset import get_dataset
from util import DynamicEUA_MCF, DynamicEUA_Random, DynamicEUA_Fuzzy, DynamicEUA_Greedy_capacity, \
    DynamicEUA_Greedy_distance


def test_method(method, user_num_per_sec, test_loader, total_sec, method_name, batch_size, data_size):
    """Test a specific allocation method."""
    seed_torch()

    test_R_list = []
    test_user_allocated_props_list = []
    test_server_costs_list = []
    test_capacity_used_props_list = []
    test_user_experience_list = []
    test_server_props_list = []
    for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
        server_seq, user_seq, masks = server_seq.cpu().numpy(), user_seq.cpu().numpy(), masks.cpu().numpy()
        this_batch_size = user_seq.shape[0]

        reward = []
        user_allocated_props = []
        server_costs = []
        capacity_used_props = []
        user_experience = []
        server_props = []

        for i in range(this_batch_size):
            servers = server_seq[i]
            users = user_seq[i]
            user_masks = masks[i]

            reward1, indicators = method(servers, users, user_masks, user_num_per_sec, total_sec)
            user_allocated_prop, server_cost, capacity_used_prop, user_experience1, server_prop = indicators
            reward.append(reward1)
            user_allocated_props.append(user_allocated_prop)
            server_costs.append(server_cost)
            capacity_used_props.append(capacity_used_prop)
            user_experience.append(user_experience1)
            server_props.append(server_prop)

        reward = np.array(reward)
        user_allocated_props = np.array(user_allocated_props)
        server_costs = np.array(server_costs)
        capacity_used_props = np.array(capacity_used_props)
        user_experience = np.array(user_experience)
        server_props = np.array(server_props)

        print(
            method_name + '\t[{}/{} ({:.1f}%)]    R:{:.6f}    user_props: {:.6f}    '
                          'server_costs: {:.6f}    capacity_props: {:.6f}    user_experience: {:.6f}    '
                          'server_props: {:.6f}'
            .format(
                min((batch_idx + 1) * batch_size, data_size['test']),
                data_size['test'],
                100. * (batch_idx + 1) / len(test_loader),
                np.mean(reward),
                np.mean(user_allocated_props),
                np.mean(server_costs),
                np.mean(capacity_used_props),
                np.mean(user_experience),
                np.mean(server_props)
            ))

        test_R_list.append(reward)
        test_user_allocated_props_list.append(user_allocated_props)
        test_server_costs_list.append(server_costs)
        test_capacity_used_props_list.append(capacity_used_props)
        test_user_experience_list.append(user_experience)
        test_server_props_list.append(server_props)

    test_R_list = np.concatenate(test_R_list)
    test_user_allocated_props_list = np.concatenate(test_user_allocated_props_list)
    test_server_costs_list = np.concatenate(test_server_costs_list)
    test_capacity_used_props_list = np.concatenate(test_capacity_used_props_list)
    test_user_experience_list = np.concatenate(test_user_experience_list)
    test_server_props_list = np.concatenate(test_server_props_list)

    test_r = np.mean(test_R_list)
    test_user_alloc = np.mean(test_user_allocated_props_list)
    test_server_cos = np.mean(test_server_costs_list)
    test_capacity_use = np.mean(test_capacity_used_props_list)
    test_user_exper = np.mean(test_user_experience_list)
    test_server_props = np.mean(test_server_props_list)

    print('')
    print(method_name + '\tTest {} \tR: {:.6f}    user_props: {:.6f}    server_costs: {:.6f}    '
                        'capacity_props: {:.6f}    user_experience: {:.6f}    server_props: {:.6f}'
          .format(method_name, test_r, test_user_alloc, test_server_cos,
                  test_capacity_use, test_user_exper, test_server_props))
    print('')

    return test_r, test_user_alloc, test_server_cos, test_capacity_use, test_user_exper, test_server_props


def test_model(model, user_num_per_sec, test_loader, device, total_sec, env_config, model_config, data_size):
    """Test a trained model."""
    model.eval()
    model.policy = 'greedy'
    env = DynamicEUAEnv(None, total_sec, env_config['capacity_reward_rate'],
                        user_stay_batch_same=True)
    if model_config['server_embedding_type'] == 'gnn':
        graph = create_graph_from_server_list(test_loader.dataset.servers_tensor, device)
    else:
        graph = None

    # Randomly distribute `user_num` users into `total_sec` time periods using a uniform distribution
    # Keep the allocation consistent within a batch to save resources and reduce computation
    # Generate `num_users` random numbers in the range [0, total_sec-1], representing the users' time periods
    user_time_periods = np.random.randint(0, total_sec, total_sec * user_num_per_sec)
    # Use `np.bincount` to count the number of users in each time period
    user_num_per_sec1 = np.bincount(user_time_periods, minlength=total_sec)

    logger.info('')
    if need_test_method:
        for method_name in method_names:
            results[method_name] = test_method(methods[method_name], user_num_per_sec1, test_loader,
                                               total_sec, method_name, batch_size, data_size)
        for method_name in method_names:
            logger.info('{:<8}\tR: {:.6f}    user_props: {:.6f}    server_costs: {:.6f}    '
                        'capacity_props: {:.6f}    user_experience: {:.6f}    '
                        'server_props: {:.6f}'.format(method_name.split('_')[-1],
                                                      *results[method_name]))

    if need_test_model:
        model_file_name = model_config['continue_model_filename']
        seed_torch()
        model = AttentionNet(6, 6, hidden_dim=model_config['hidden_dim'], device=device,
                             exploration_c=model_config['exploration_c'],
                             server_embedding_type=model_config['server_embedding_type'],
                             total_sec=data_config['total_sec'])
        checkpoint = torch.load(model_file_name, map_location='cpu')
        if model_file_name.endswith('.mdl'):
            model.load_state_dict(checkpoint)
        elif model_file_name.endswith('.pt'):
            model.load_state_dict(checkpoint['model'])
        else:
            raise ValueError('model_file_name should end with .mdl or .pt')
        model = model.to(device)
        results['model'] = test_model(model, user_num_per_sec1, test_loader, device,
                                      total_sec, env_config, model_config, data_size)
        logger.info('{:<8}\tR: {:.6f}    user_props: {:.6f}    server_costs: {:.6f}    '
                    'capacity_props: {:.6f}    user_experience: {:.6f}    '
                    'server_props: {:.6f}'.format('model', *results['model']))

    # Final summary output
    logger.info('')
    if need_test_method:
        for method_name in method_names:
            logger.info('{:<8}\tR: {:.6f}    user_props: {:.6f}    server_costs: {:.6f}    '
                        'capacity_props: {:.6f}    user_experience: {:.6f}    '
                        'server_props: {:.6f}'.format(method_name.split('_')[-1],
                                                      *results[method_name]))
    if need_test_model:
        logger.info('{:<8}\tR: {:.6f}    user_props: {:.6f}    server_costs: {:.6f}    '
                    'capacity_props: {:.6f}    user_experience: {:.6f}    '
                    'server_props: {:.6f}'.format('model', *results['model']))
