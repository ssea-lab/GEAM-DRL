import os
import numpy as np
import torch
from torch.utils.data import Dataset

from data.data_generator import init_server, init_users_list_by_positions_and_mask, init_user_position_and_mask_by_servers
from util.utils import save_dataset


class EuaDataset(Dataset):
    """
    A dataset class for Edge User Allocation (EUA).
    """

    def __init__(self, servers, users_list, users_masks_list, device):
        """
        :param servers: Server information (positions and capacities)
        :param users_list: List of user sequences
        :param users_masks_list: List of user mask sequences
        :param device: The computing device (CPU/GPU)
        """
        self.servers = servers
        self.users_list = users_list
        self.users_masks_list = users_masks_list
        self.servers_tensor = torch.tensor(servers, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        """Return the number of user sequences."""
        return len(self.users_list)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset.

        :param index: Index of the sample
        :return: Server tensor, user sequence, and mask sequence
        """
        user_seq = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
        return self.servers_tensor, user_seq, mask_seq


# Uncomment this class if user number per second needs to be included in the dataset
# class EuaDataset(Dataset):
#     def __init__(self, servers, users_list, users_masks_list, users_nums_per_sec, device):
#         """
#         :param servers: Server information (positions and capacities)
#         :param users_list: List of user sequences
#         :param users_masks_list: List of user mask sequences
#         :param users_nums_per_sec: Number of users per second in each sequence
#         :param device: The computing device (CPU/GPU)
#         """
#         self.servers = servers
#         self.users_list = users_list
#         self.users_masks_list = users_masks_list
#         self.users_nums_per_sec = users_nums_per_sec
#         self.servers_tensor = torch.tensor(servers, dtype=torch.float32, device=device)
#         self.device = device
#
#     def __len__(self):
#         """Return the number of user sequences."""
#         return len(self.users_list)
#
#     def __getitem__(self, index):
#         """
#         Retrieve a sample from the dataset.
#
#         :param index: Index of the sample
#         :return: Server tensor, user sequence, mask sequence, and user number per second
#         """
#         user_seq = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
#         mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
#         user_num_per_sec = self.users_nums_per_sec[index]
#         return self.servers_tensor, user_seq, mask_seq, user_num_per_sec


def get_dataset(x_end, y_end, miu, sigma, total_sec, user_num_per_sec, user_stay_miu, user_stay_sigma,
                data_size: {}, max_cov, device, dir_name):
    """
    Retrieve or generate a dataset.

    :param x_end: X-axis range limit
    :param y_end: Y-axis range limit
    :param miu: Mean server capacity
    :param sigma: Standard deviation of server capacity
    :param total_sec: Total duration for user sequence generation
    :param user_num_per_sec: Average number of users generated per second
    :param user_stay_miu: Mean user stay duration
    :param user_stay_sigma: Standard deviation of user stay duration
    :param data_size: Dictionary specifying the number of samples for each dataset type (train, valid, test)
    :param max_cov: Maximum coverage radius of servers
    :param device: Computing device (CPU/GPU)
    :param dir_name: Directory to store the dataset
    :return: A dictionary containing train, validation, and test datasets
    """
    dataset_dir_name = os.path.join(dir_name,
                                    "dataset/server_" + str(x_end) + "_" + str(y_end)
                                    + "_miu_" + str(miu) + "_sigma_" + str(sigma))
    server_file_name = "server_" + str(x_end) + "_" + str(y_end) + "_miu_" + str(miu) + "_sigma_" + str(sigma)
    server_path = os.path.join(dataset_dir_name, server_file_name) + '.npy'
    user_position_and_mask_file_name = "user_position_and_mask_" + str(x_end) + "_" + str(y_end) + "_num_" + str(10000)
    user_position_and_mask_path = os.path.join(dataset_dir_name, user_position_and_mask_file_name) + '.npz'

    # Load pre-existing server and user position/mask data if available
    if os.path.exists(server_path) and os.path.exists(user_position_and_mask_path):
        servers = np.load(server_path)
        user_position_and_mask = np.load(user_position_and_mask_path)
        print("Successfully loaded server and user position/mask data.")
    else:
        print("Server data not found, regenerating...")
        os.makedirs(dataset_dir_name, exist_ok=True)
        servers = init_server(0, x_end, 0, y_end, max_cov, miu, sigma)
        np.save(server_path, servers)
        print("Server data saved. Generating user positions and masks...")
        # Generate 10,000 user positions and masks and save them for efficient dataset generation
        user_position_and_mask = init_user_position_and_mask_by_servers(servers, 10000, max_cov)
        save_dataset(user_position_and_mask_path, **user_position_and_mask)
        print("User position and mask data saved.")

    set_types = data_size.keys()
    datasets = {}
    for set_type in set_types:
        if set_type not in ('train', 'valid', 'test'):
            raise NotImplementedError(f"Dataset type '{set_type}' is not supported.")
        
        filename = (set_type + "_user_num_per_sec_" + str(user_num_per_sec) + "_user_stay_miu_"
                    + str(user_stay_miu) + "_size_" + str(data_size[set_type]))
        path = os.path.join(dataset_dir_name, filename) + '.npz'

        # Load existing dataset if available; otherwise, generate and save it
        if os.path.exists(path):
            print(f"Loading {set_type} dataset...")
            data = np.load(path)
        else:
            print(f"{set_type} dataset not found, regenerating {path}...")
            data = init_users_list_by_positions_and_mask(user_position_and_mask, data_size[set_type],
                                                         total_sec, user_num_per_sec, user_stay_miu, user_stay_sigma)
            save_dataset(path, **data)
        
        datasets[set_type] = EuaDataset(servers, **data, device=device)

    return datasets


def shuffle_dataset(test_set):
    """
    Shuffle the user sequences in the dataset.

    :param test_set: The dataset to shuffle
    :return: A new shuffled dataset
    """
    new_users = []
    new_masks = []
    for i in range(len(test_set)):
        x = zip(test_set.users_list[i], test_set.users_masks_list[i])
        x = list(x)
        np.random.shuffle(x)
        new_user, new_mask = zip(*x)
        new_users.append(new_user)
        new_masks.append(new_mask)
    
    new_users_array = np.stack(new_users)
    new_masks_array = np.stack(new_masks)
    return EuaDataset(test_set.servers, new_users_array, new_masks_array, test_set.device)
