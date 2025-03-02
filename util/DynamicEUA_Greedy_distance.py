import copy
import time
import numpy as np

from util.common_methods import calc_rewards, get_reward, can_allocate
from util.utils import mask_trans_to_list, PriorityQueue


def greedy_distance_allocate(servers, users, user_masks, user_num_per_sec, total_sec=100):
    # 记录所有用户分配所用的时间，不包括计算指标的时间
    all_time = 0
    server_num = len(servers)
    user_within_servers = mask_trans_to_list(user_masks, server_num)

    # 每个服务器分配到的用户数量
    server_allocate_num = np.zeros(server_num, dtype=int)

    # 复制一份server，防止改变工作负载源数据
    tmp_server_capacity = np.array(copy.deepcopy([server[2:] for server in servers]))
    # 表示当前用户的索引
    i = -1
    indicators = []

    # 维护一个优先队列，记录每个用户退出的时间
    # 优先队列的元素是一个元组，(退出时间，用户索引，服务器索引)
    exit_queue = PriorityQueue()
    # 为每一个用户分配一个服务器

    # 以总秒数作为循环
    for current_time in range(total_sec):
        # 在进入这一秒之前，先把这一秒要退出的用户减掉
        if current_time > 0:
            while not exit_queue.is_empty() and exit_queue.top()[0] <= current_time:
                _, user_idx, server_idx = exit_queue.pop()
                # 服务器分配矩阵减一
                server_allocate_num[server_idx] -= 1
                # 服务器分配矩阵中的最后一个表示云端，只是占位符，加了减了随便，不用管
                # 服务器容量加回来
                tmp_server_capacity[server_idx] += users[user_idx, 2:-1]

        current_user_len = user_num_per_sec[current_time]
        # 记录这一秒内分配的用户的分配情况
        user_allocate_list = -np.ones(current_user_len, dtype=int)

        if current_user_len == 0:
            continue

        # 记录当前时间
        start_time = time.time()

        for user_i in range(current_user_len):
            i += 1
            user = users[i]
            workload = user[2: -1]

            # 在可用server中寻找距离最近的
            distance = 100000
            final_server_id = -1
            for server_id in user_within_servers[i]:
                capacity = tmp_server_capacity[server_id]
                if can_allocate(workload, capacity):
                    # 计算距离
                    user_position = user[:2]
                    server_position = servers[server_id][:2]
                    dis = np.linalg.norm(user_position - server_position)
                    if dis < distance:
                        distance = dis
                        final_server_id = server_id
            if final_server_id != -1:
                tmp_server_capacity[final_server_id] -= workload
                user_allocate_list[user_i] = final_server_id
                server_allocate_num[final_server_id] += 1
                # 记录该用户退出的时间，放到优先队列里
                exit_time = current_time + user[-1]
                exit_queue.push((exit_time, i, final_server_id))
            # 如果for循环结束都没有找到合适的服务器，那么这个用户分配列表就是-1，不用管

        # 记录时间
        all_time += time.time() - start_time

        # 开始计算指标
        current_users_positions = users[i + 1 - current_user_len: i + 1, :2]
        reward = calc_rewards(current_users_positions, user_allocate_list, current_user_len,
                              server_allocate_num > 0,
                              server_num, servers[:, :], tmp_server_capacity)

        if len(indicators) == 0:
            indicators = reward
        else:
            indicators = [indicators[j] + reward[j] for j in range(len(indicators))]
    # 计算总奖励
    indicators = [indicators[j] / total_sec for j in range(len(indicators))]

    indicators.append(all_time)

    return get_reward(*indicators[:4]), indicators
