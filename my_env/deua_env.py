import numpy as np
import torch

from util.utils import PriorityQueue


def can_allocate(workload: torch.Tensor, capacity: torch.Tensor):
    """
    计算能不能分配并返回分配情况
    :param workload: (batch, 4)
    :param capacity: (batch, 4)
    :return:
    """
    # (batch, 4)
    bools = capacity >= workload
    # (batch)，bool值
    return torch.all(bools, dim=1)


class DynamicEUAEnv:

    def __init__(self, user_num_per_sec, total_sec, capacity_reward_rate, user_stay_batch_same):
        self.user_num_per_sec = user_num_per_sec
        self.total_sec = total_sec
        self.capacity_reward_rate = capacity_reward_rate
        self.user_stay_batch_same = user_stay_batch_same

    def get_exit_queue(self, batch_size):
        # 如果一个batch内保持一致，就返回一个优先队列
        if self.user_stay_batch_same:
            return PriorityQueue()
        else:
            return [PriorityQueue() for _ in range(batch_size)]

    def push_exit_queue(self, exit_queue, exit_time, i, idx):
        if self.user_stay_batch_same:
            # 如果一个batch内保持一致，就直接放到优先队列里
            # i表示用户索引，是一个数，idx表示服务器索引，是(batch_size, user_len)，idx有可能是-1
            exit_queue.push((exit_time[0].item(), i, idx))
        else:
            batch_size = exit_time.size(0)
            for j in range(batch_size):
                # 如果分配的服务器不是-1，就放到优先队列里
                if idx[j].item() != -1:
                    exit_queue[j].push((exit_time[j].item(), i, idx[j].item()))
        return exit_queue

    def release_users(self, current_time, user_input_seq_with_stay,
                      server_allocate_mat, tmp_server_capacity, exit_queue):
        batch_size = user_input_seq_with_stay.size(0)
        if self.user_stay_batch_same:
            batch_range = torch.arange(batch_size)
            # 如果一个batch内保持一致，就直接处理
            while not exit_queue.is_empty() and exit_queue.top()[0] <= current_time:
                _, user_idx, server_idx = exit_queue.pop()
                # 服务器分配矩阵减一
                server_allocate_mat[batch_range, server_idx] -= 1
                # 服务器分配矩阵中的最后一个表示云端，只是占位符，加了减了随便，不用管
                # 服务器容量加回来
                tmp_server_capacity = torch.cat(
                    (tmp_server_capacity, torch.zeros(batch_size, 1, 4, device=tmp_server_capacity.device)), dim=1)
                tmp_server_capacity[batch_range, server_idx] += user_input_seq_with_stay[batch_range, user_idx, 2:-1]
                tmp_server_capacity = tmp_server_capacity[:, :-1]
        else:
            for j in range(batch_size):
                while not exit_queue[j].is_empty() and exit_queue[j].top()[0] <= current_time:
                    _, user_idx, server_idx = exit_queue[j].pop()
                    # 服务器分配矩阵减一
                    server_allocate_mat[j, server_idx] -= 1
                    # 服务器分配矩阵中的最后一个表示云端，只是占位符，加了减了随便，不用管
                    # 服务器容量加回来
                    tmp_server_capacity[j, server_idx] += user_input_seq_with_stay[j, user_idx, 2:-1]
        return server_allocate_mat, tmp_server_capacity

    def get_user_num_per_sec_list(self):
        # 把user_num个用户随机分配到total_sec个时间段，使用均匀分布
        # 在一个batch内保持一致，节省资源，减小计算量
        # 生成num_users个0到num_time_periods-1的随机数，表示用户所在的时间段
        user_time_periods = np.random.randint(0, self.total_sec, self.total_sec * self.user_num_per_sec)
        # 使用np.bincount统计每个时间段的用户数量
        user_num_per_sec = np.bincount(user_time_periods, minlength=self.total_sec)
        return user_num_per_sec

    @staticmethod
    def update_server_capacity(server_id, tmp_server_capacity, user_workload):
        batch_size = server_id.size(0)
        # 取出一个batch里所有第j个用户选择的服务器
        index_tensor = server_id.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, 4)  # 4个资源维度
        j_th_server_capacity = torch.gather(tmp_server_capacity, dim=1, index=index_tensor).squeeze(1)
        # (batch_size)的True，False矩阵
        can_be_allocated = can_allocate(user_workload, j_th_server_capacity)
        # 如果不能分配容量就不减
        mask = can_be_allocated.unsqueeze(-1).expand(batch_size, 4)
        # 切片时指定batch和server_id对应关系
        batch_range = torch.arange(batch_size)
        # 服务器减去相应容量
        tmp_server_capacity[batch_range, server_id] -= user_workload * mask
        # 记录服务器分配情况，即server_id和mask的内积
        server_id = torch.masked_fill(server_id, mask=~can_be_allocated, value=-1)
        return tmp_server_capacity, server_id

    @staticmethod
    def calc_user_experience_by_distance(distance):
        return 1 + 1 / ((distance + 1) ** 2)

    def calc_rewards(self, current_user_positions, user_allocate_list, user_len, server_allocate_mat, server_len,
                     original_servers, batch_size, tmp_server_capacity):
        # 目前user_allocate_list是(batch_size, user_len)
        # 计算每个分配的用户数，即不是-1的个数，(batch_size)
        user_allocate_num = torch.sum(user_allocate_list != -1, dim=1)
        user_allocated_props = user_allocate_num.float() / user_len
        # 计算服务器的租用率
        server_used_num = torch.sum(server_allocate_mat[:, :-1], dim=1)
        server_used_props = server_used_num.float() / server_len
        # 计算服务器租用成本
        # 租用成本就是所有开启的服务器的成本之和
        # 每个服务器的成本是服务器的资源量之和
        server_allocated_flag = server_allocate_mat[:, :-1].unsqueeze(-1).expand(batch_size, server_len, 4)
        # (batch_size, server_len, 4)
        original_servers_capacity = original_servers[:, :, 2:]
        used_original_server = original_servers_capacity.masked_fill(~server_allocated_flag.bool(), value=0)
        cost = self.calc_cost(used_original_server)

        # 计算用户体验
        # 计算每个用户的位置和分配的服务器的位置的距离
        original_servers_position = original_servers[:, :, :2]  # (batch_size, server_len, 2)
        # 在原始服务器位置中添加一个服务器在-1下标，表示云端，坐标为0,0，不参与运算
        original_servers_position = torch.cat((original_servers_position,
                                               torch.zeros(batch_size, 1, 2, device=original_servers.device)), dim=1)
        # 把current_user中分配下标为-1的用户mask掉，从而能不参与运算
        mask = user_allocate_list != -1
        current_user_positions = current_user_positions.masked_fill(~mask.view(batch_size, -1, 1).repeat(1, 1, 2),
                                                                    torch.inf)
        # 根据用户分配的服务器索引获取对应的服务器坐标
        # 把负数的下标转换为正数的下标，即-1转换为num_servers
        user_allocate_list = user_allocate_list.remainder(server_len + 1)
        # (batch_size, user_len, 2)
        user_allocated_server_position = torch.gather(original_servers_position, 1,
                                                      user_allocate_list.view(batch_size, -1, 1).repeat(1, 1, 2))
        # (batch_size, user_len, 2)
        distances = torch.norm(current_user_positions - user_allocated_server_position, dim=2)
        experience = self.calc_user_experience_by_distance(distances)
        experience[~mask] = 0

        user_experience = torch.mean(experience, dim=1)

        # 已使用的服务器的资源利用率
        # (batch_size, server_len, 4)
        servers_remain_capacity = tmp_server_capacity.masked_fill(~server_allocated_flag.bool(), value=0)
        # 对于每个维度的资源求和，得到的结果应该是(batch_size, 4)，被压缩的维度是1
        sum_all_capacity = torch.sum(used_original_server, dim=1)
        sum_remain_capacity = torch.sum(servers_remain_capacity, dim=1)
        # 对于每个维度的资源求资源利用率
        every_capacity_remain_props = torch.div(sum_remain_capacity, sum_all_capacity)
        mean_capacity_remain_props = torch.mean(every_capacity_remain_props, dim=1)
        capacity_used_props = 1 - mean_capacity_remain_props
        return user_allocated_props, cost, capacity_used_props, user_experience, server_used_props

    def get_reward(self, user_allocated_props, server_used_props, capacity_used_props, user_experience):
        return (1 - self.capacity_reward_rate) * user_experience + self.capacity_reward_rate * capacity_used_props

    @staticmethod
    def calc_cost(used_original_server):
        # 计算服务器租用成本
        # 租用成本就是所有开启的服务器的成本之和
        # 每个服务器的成本是服务器的资源量之和
        # (batch_size, server_len)
        every_server_cost = torch.sum(used_original_server, dim=2) / 100
        # (batch_size)
        all_server_cost = torch.sum(every_server_cost, dim=1)
        return all_server_cost
