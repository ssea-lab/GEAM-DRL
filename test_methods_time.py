import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from my_env.deua_env import DynamicEUAEnv
from nets.attention_net_to_calc_time import AttentionNet
from util.torch_utils import seed_torch, create_graph_from_server_list
from data.eua_dataset import get_dataset
from util import DynamicEUA_MCF, DynamicEUA_Random, DynamicEUA_Fuzzy, DynamicEUA_Greedy_capacity, \
    DynamicEUA_Greedy_distance


def test_method(method, user_num_per_sec, test_loader, total_sec):
    """
    Test a given allocation method.

    :param method: Allocation method to test
    :param user_num_per_sec: Number of users per second
    :param test_loader: Test data loader
    :param total_sec: Total duration (in seconds)
    :return: Total time taken for allocation
    """
    seed_torch()
    for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
        server_seq, user_seq, masks = server_seq.cpu().numpy(), user_seq.cpu().numpy(), masks.cpu().numpy()
        this_batch_size = user_seq.shape[0]

        for i in range(this_batch_size):
            servers = server_seq[i]
            users = user_seq[i]
            user_masks = masks[i]

            _, indicators = method(servers, users, user_masks, user_num_per_sec, total_sec)
            _, _, _, _, _, all_time = indicators

            return all_time


def test_model(model, user_num_per_sec, test_loader, device, total_sec, env_config, model_config):
    """
    Test a trained model.

    :param model: Trained model to test
    :param user_num_per_sec: Number of users per second
    :param test_loader: Test data loader
    :param device: Computing device (CPU/GPU)
    :param total_sec: Total duration (in seconds)
    :param env_config: Environment configuration
    :param model_config: Model configuration
    :return: Total time taken for allocation
    """
    with torch.no_grad():
        model.eval()
        model.policy = 'greedy'
        env = DynamicEUAEnv(None, total_sec, env_config['capacity_reward_rate'],
                            user_stay_batch_same=True)
        if model_config['server_embedding_type'] == 'gnn':
            graph = create_graph_from_server_list(test_loader.dataset.servers_tensor, device)
        else:
            graph = None

        for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
            server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

            # During testing, the user's stay duration should vary to remain consistent with other methods
            env.user_stay_batch_same = False
            all_time = model(env, user_seq, server_seq, masks, graph, user_num_per_sec)

            return all_time


def load_model(model, model_path):
    """
    Load a pre-trained model.

    :param model: Model instance
    :param model_path: Path to the saved model
    :return: Loaded model
    """
    model.load_state_dict(torch.load(model_path))
    return model


def main(need_test_method, need_test_model):
    """
    Main function for testing allocation methods and trained models.

    :param need_test_method: Boolean flag to test allocation methods
    :param need_test_model: Boolean flag to test the trained model
    """
    seed_torch()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_config, data_config, model_config, env_config = (config['train'], config['data'],
                                                           config['model'], config['env'])
    device = 'cpu'
    data_size = {'test': 1}
    user_num_per_sec_config = 100
    dataset = get_dataset(data_config['x_end'], data_config['y_end'], data_config['miu'], data_config['sigma'],
                          data_config['total_sec'], user_num_per_sec_config,
                          data_config['user_stay_miu'], data_config['user_stay_sigma'],
                          data_size, data_config['max_cov'], device, train_config['dir_name'])
    batch_size = 1

    total_sec = data_config['total_sec']

    test_loader = DataLoader(dataset=dataset['test'], batch_size=batch_size, shuffle=False)

    methods = {"greedy_capacity": DynamicEUA_Greedy_capacity.greedy_capacity_allocate,
               "greedy_distance": DynamicEUA_Greedy_distance.greedy_distance_allocate,
               "random": DynamicEUA_Random.random_allocate,
               "fuzzy": DynamicEUA_Fuzzy.fuzzy_allocate,
               "mcf": DynamicEUA_MCF.mcf_allocate,
               }
    method_names = ["greedy_capacity", "greedy_distance", "random", "fuzzy", "mcf"]

    # Randomly distribute `user_num` users into `total_sec` time periods using a uniform distribution
    # Keep the allocation consistent within a batch to save resources and reduce computation
    # Generate `num_users` random numbers in the range [0, total_sec-1], representing the users' time periods
    user_time_periods = np.random.randint(0, total_sec, total_sec * user_num_per_sec_config)
    # Use `np.bincount` to count the number of users in each time period
    user_num_per_sec1 = np.bincount(user_time_periods, minlength=total_sec)

    user_num = user_num_per_sec_config * total_sec

    if need_test_method:
        for method_name in method_names:
            print(method_name)
            all_time = test_method(methods[method_name], user_num_per_sec1, test_loader, total_sec)
            # Convert time units to milliseconds
            print("all_time: ", all_time * 1000)
            print("average_time: ", all_time / user_num * 1000)

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
        print('model')
        all_time = test_model(model, user_num_per_sec1, test_loader, device, total_sec, env_config, model_config)
        # Convert time units to milliseconds
        print("all_time: ", all_time * 1000)
        print("average_time: ", all_time / user_num * 1000)


if __name__ == '__main__':
    need_test_method1 = True
    need_test_model1 = True
    main(need_test_method1, need_test_model1)
