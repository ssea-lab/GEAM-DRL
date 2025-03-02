import numpy as np


def can_allocate(workload, capacity):
    return np.all(workload <= capacity)


def get_reward(reward1, reward2, reward3, reward4):
    return 0.5 * reward4 + 0.5 * reward3


def calc_user_experience_by_distance(distance):
    return 1 + 1 / ((distance + 1) ** 2)


def calc_rewards(current_user_positions, user_allocate_list, user_len, server_allocate_num, server_len,
                 original_servers, tmp_server_capacity):
    # 目前user_allocate_list是(user_len)
    # 计算每个分配的用户数，即不是-1的个数，(batch_size)
    user_allocate_num = np.sum(user_allocate_list != -1)
    user_allocated_props = user_allocate_num / user_len
    # 计算服务器的租用率
    server_used_num = np.sum(server_allocate_num > 0)
    server_used_props = server_used_num / server_len
    # 计算服务器租用成本
    # 租用成本就是所有开启的服务器的成本之和
    # 每个服务器的成本是服务器的资源量之和
    server_allocate_mat = server_allocate_num > 0
    used_original_server = original_servers[server_allocate_mat]
    original_servers_capacity = used_original_server[:, 2:]
    cost = calc_cost(used_original_server)

    # 计算用户体验
    # 计算每个用户的位置和分配的服务器的位置的距离
    original_servers_position = original_servers[:, :2]  # (server_len, 2)
    # 把current_user中分配下标为-1的用户mask掉，从而能不参与运算
    mask = user_allocate_list != -1
    current_user_positions = current_user_positions[mask]

    # 根据用户分配的服务器索引获取对应的服务器坐标
    # (good_user_len, 2)
    user_allocated_server_position = original_servers_position[user_allocate_list[mask]]
    # 计算每个用户与对应服务器之间的距离 (good_user_len, 2)
    distances = np.linalg.norm(current_user_positions - user_allocated_server_position, axis=1)
    experience = calc_user_experience_by_distance(distances)

    # 不可以用mean，因为有的用户没有分配到服务器，所以要用sum
    user_experience = np.sum(experience) / user_len

    # 已使用的服务器的资源利用率
    servers_remain_capacity = tmp_server_capacity[server_allocate_mat]
    sum_all_capacity = np.sum(original_servers_capacity, axis=0)
    sum_remain_capacity = np.sum(servers_remain_capacity, axis=0)
    # 对于每个维度的资源求资源利用率
    every_capacity_remain_props = np.divide(sum_remain_capacity, sum_all_capacity)
    mean_capacity_remain_props = np.mean(every_capacity_remain_props, axis=0)
    capacity_used_props = 1 - mean_capacity_remain_props

    return user_allocated_props, cost, capacity_used_props, user_experience, server_used_props


def calc_cost(used_original_server):
    # 计算服务器租用成本
    # 租用成本就是所有开启的服务器的成本之和
    # 每个服务器的成本是服务器的资源量之和
    # (server_len, 4) -> (server_len, 1) -> (1, 1)
    every_server_cost = np.sum(used_original_server, axis=1) / 100
    all_server_cost = np.sum(every_server_cost)
    return all_server_cost
