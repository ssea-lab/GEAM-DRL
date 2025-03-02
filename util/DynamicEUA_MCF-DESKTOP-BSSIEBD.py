import copy
import numpy as np

from util.common_methods import calc_rewards, get_reward, can_allocate
from util.utils import mask_trans_to_list, PriorityQueue


def mcf_allocate(servers, users, user_masks, user_num_per_sec, total_sec=100):
    server_num = len(servers)
    user_within_servers = mask_trans_to_list(user_masks, server_num)

    # 每个服务器分配到的用户数量
    server_allocate_num = np.zeros(server_num, dtype=int)

    # 复制一份server，防止改变工作负载源数据
    tmp_server_capacity = np.array(copy.deepcopy([server[2:] for server in servers]))
    # 表示当前用户的索引
    i = -1
    reward1 = 0
    reward2 = 0
    reward3 = 0
    reward4 = 0

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

        for user_i in range(current_user_len):
            i += 1
            user = users[i]
            workload = user[2: -1]
            # 先过滤为已经激活的
            this_user_s_active_server = []
            other_servers = []
            for server_id in user_within_servers[i]:
                if server_allocate_num[server_id] > 0:
                    this_user_s_active_server.append(server_id)
                else:
                    other_servers.append(server_id)

            # 在可用server中寻找剩余容量最多的
            max_remain_capacity = -1
            final_server_id = -1
            for server_id in this_user_s_active_server:
                capacity = tmp_server_capacity[server_id]
                if can_allocate(workload, capacity):
                    # 计算总剩余容量
                    remain_capacity = sum(capacity) - sum(workload)
                    if remain_capacity > max_remain_capacity:
                        max_remain_capacity = remain_capacity
                        final_server_id = server_id
            # 如果没找到合适的服务器，那么在其他服务器中找
            if final_server_id == -1:
                for server_id in other_servers:
                    capacity = tmp_server_capacity[server_id]
                    if can_allocate(workload, capacity):
                        # 计算总剩余容量
                        remain_capacity = sum(capacity) - sum(workload)
                        if remain_capacity > max_remain_capacity:
                            max_remain_capacity = remain_capacity
                            final_server_id = server_id
            if final_server_id != -1:
                tmp_server_capacity[final_server_id] -= workload
                user_allocate_list[user_i] = final_server_id
                server_allocate_num[final_server_id] += 1
                # 记录该用户退出的时间，放到优先队列里
                exit_time = current_time + user[-1]
                exit_queue.push((exit_time, i, final_server_id))
            # 如果for循环结束都没有找到合适的服务器，那么这个用户分配列表就是-1，不用管

        # 开始计算指标
        current_users_positions = users[i + 1 - current_user_len: i + 1, :2]
        user_allocated_props, server_used_props, capacity_used_props, user_experience = \
            calc_rewards(current_users_positions, user_allocate_list, current_user_len, server_allocate_num > 0,
                         server_num, servers[:, :], tmp_server_capacity)

        reward1 += user_allocated_props
        reward2 += server_used_props
        reward3 += capacity_used_props
        reward4 += user_experience

    # 计算平均值
    reward1 /= total_sec
    reward2 /= total_sec
    reward3 /= total_sec
    reward4 /= total_sec

    return get_reward(reward1, reward2, reward3, reward4), reward1, reward2, reward3, reward4
