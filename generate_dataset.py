import yaml

from data.eua_dataset import get_dataset
from util.torch_utils import seed_torch


def main_get_dataset():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dir_name = config['train']['dir_name']
    total_sec = config['data']['total_sec']
    user_num_per_sec = 10
    user_stay_miu = 30
    user_stay_sigma = 5
    max_cov = 1.5
    miu = 35
    sigma = 10
    data_size = {
        'train': 10000,
        'valid': 1000,
        'test': 1000
    }

    seed_torch(42)
    get_dataset(0.5, 1, miu, sigma, total_sec, user_num_per_sec, user_stay_miu, user_stay_sigma,
                data_size, max_cov, 'cpu', dir_name)


if __name__ == '__main__':
    main_get_dataset()
