import copy
import time

import numpy as np

from util.common_methods import calc_rewards, get_reward, can_allocate
from util.utils import mask_trans_to_list, PriorityQueue

EL, VL, L, M, H, VH, EH = 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1
omega_dic = {'ML': {"SL": EL, "SM": VL, "SH": VL},
             'MM': {"SL": M, "SM": L, "SH": VL},
             'MH': {"SL": EH, "SM": VH, "SH": H}}
gamma = 1.5


def get_fuzzy_weight(mu, std):
    if mu <= 0.09:
        a = 'ML'
    elif 0.09 < mu <= 0.22:
        a = 'MM'
    else:
        a = 'MH'
    if std <= 0.03:
        b = 'SL'
    elif 0.03 < std <= 0.12:
        b = 'SM'
    else:
        b = 'SH'
    return omega_dic[a][b]


def fuzzy_allocate(servers, users, user_masks, user_num_per_sec, total_sec=100):
    # 记录所有用户分配所用的时间，不包括计算指标的时间
    all_time = 0
    server_num = len(servers)
    user_within_servers = mask_trans_to_list(user_masks, server_num)

    # 每个服务器分配到的用户数量
    server_allocate_num = np.zeros(server_num, dtype=int)

    # 复制一份server，防止改变工作负载源数据
    tmp_server_capacity = np.array(copy.deepcopy([server[2:] for server in servers]))
    # 记录服务器释放时间
    server_release_time = np.zeros(server_num, dtype=int)
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

            # 计算所有服务器的资源利用率
            capacity_used_props = 1 - tmp_server_capacity / servers[:, 2:]
            capacity_used_props = np.mean(capacity_used_props, axis=1)
            # 计算资源利用率的均值和方差
            mu = np.mean(capacity_used_props)
            std = np.std(capacity_used_props)

            # 开始遍历服务器找最高分
            final_server_ids = []
            C = []
            B = []
            for server_id in user_within_servers[i]:
                capacity = tmp_server_capacity[server_id]
                if can_allocate(workload, capacity):
                    # 使用模糊控制机制计算得分
                    final_server_ids.append(server_id)
                    # 首先计算整合分数c
                    # 服务器还没开启，那预计释放时间就是0
                    zi = server_release_time[server_id]
                    t = current_time  # 当前时间
                    vj = user[-1]  # 需要占用的时间
                    c = abs(zi - (t + vj))  # 所以t + vj是新的预计释放时间
                    if zi < t + vj:
                        c = c * gamma
                    C.append(c)
                    # 上面的代码实现了：如果开启新的服务器，c = 10 * 1.5 = 15, 否则c = 0
                    # 然后计算b，b就是这个服务器四个维度的资源利用率的平均值，b越大，服务器压力越大
                    b = capacity_used_props[server_id]
                    B.append(b)
            if final_server_ids:
                # 然后就要用模糊控制机制得到权重，从而计算分数
                omega_j = get_fuzzy_weight(mu, std)
                # B 和 C 归一化，然后计算S
                max_c, min_c = max(C), min(C)
                max_b, min_b = max(B), min(B)
                S = []
                for x in range(len(C)):
                    ci = (C[x] - min_c) / (max_c - min_c) if max_c - min_c != 0 else 0
                    bi = (B[x] - min_b) / (max_b - min_b) if max_b - min_b != 0 else 0
                    S.append(omega_j * ci + (1 - omega_j) * bi)

                final_server_id = final_server_ids[np.argmin(np.array(S))]
                tmp_server_capacity[final_server_id] -= workload
                user_allocate_list[user_i] = final_server_id
                server_allocate_num[final_server_id] += 1
                # 记录该用户退出的时间，放到优先队列里
                exit_time = current_time + user[-1]
                exit_queue.push((exit_time, i, final_server_id))
                # 更新服务器释放时间
                server_release_time[final_server_id] = max(server_release_time[final_server_id], exit_time)
                # 先看C和B到底是要干啥：
                # 首先是argmin，是为了最小化这两个指标
                # C越小，代表不用开启新服务器，所以最小化C是不开启新服务器，也就是用户整合
                # B越小，代表这个服务器越空，所以最小化B，是把用户往空的服务器上分配，偏重负载均衡和开启新服务器

                # 模糊控制的原理：
                # mu越小，std越大，都会让omega越大，也就是更偏重C的分数，C就是偏重整合用户，让用户更集中
        # 记录时间
        all_time += time.time() - start_time

        # 开始计算指标
        current_users_positions = users[i + 1 - current_user_len: i + 1, :2]
        reward = calc_rewards(current_users_positions, user_allocate_list, current_user_len, server_allocate_num > 0,
                              server_num, servers[:, :], tmp_server_capacity)

        if len(indicators) == 0:
            indicators = reward
        else:
            indicators = [indicators[j] + reward[j] for j in range(len(indicators))]
    # 计算总奖励
    indicators = [indicators[j] / total_sec for j in range(len(indicators))]

    indicators.append(all_time)

    return get_reward(*indicators[:4]), indicators
