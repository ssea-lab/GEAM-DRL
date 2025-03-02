import time

import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv
from my_env.deua_env import DynamicEUAEnv


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, 'mean')

    def forward(self, features, g):
        edge_weight = g.edata['weight']
        # 第一层图卷积
        x = self.conv1(g, features, edge_weight=edge_weight)
        return x


class UserEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(UserEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, inputs):
        return self.embedding(inputs)


class ServerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_type='linear'):
        super(ServerEncoder, self).__init__()
        if embedding_type == 'linear':
            self.embedding = nn.Linear(input_dim, hidden_dim)
        elif embedding_type == 'gnn':
            self.embedding = GNN(input_dim, hidden_dim)

    def forward(self, inputs, g=None):
        if g is not None:
            # 如果是图神经网络，要把batch_size维度和server_num维度调换顺序
            inputs = inputs.permute(1, 0, 2)
            return self.embedding(inputs, g).permute(1, 0, 2)
        return self.embedding(inputs)


class Attention(nn.Module):
    def __init__(self, hidden_dim, exploration_c=10):
        super(Attention, self).__init__()
        self.hidden_size = hidden_dim
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vt = nn.Linear(hidden_dim, 1, bias=False)
        self.exploration_c = exploration_c

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1, hidden_size)
        decoder_transform = self.W2(decoder_state)

        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts
        # log_softmax for a better numerical stability
        score = u_i.masked_fill(~mask, value=torch.log(torch.tensor(1e-45))) * self.exploration_c
        prob = torch.softmax(score, dim=-1)
        return prob


class AttentionNet(nn.Module):
    def __init__(self, user_input_dim, server_input_dim, hidden_dim, device, server_embedding_type,
                 exploration_c=10, policy='sample', total_sec=100):
        super(AttentionNet, self).__init__()
        # decoder hidden size
        self.hidden_dim = hidden_dim
        self.device = device
        self.total_sec = total_sec
        self.user_encoder = UserEncoder(user_input_dim, hidden_dim).to(device)
        self.server_encoder = ServerEncoder(server_input_dim + 1, hidden_dim, server_embedding_type).to(device)

        self.pointer = Attention(hidden_dim, exploration_c).to(device)
        self.policy = policy

    def choose_server_id(self, mask, user, static_server_seq, tmp_server_capacity, server_active, graph=None):
        """
        每一步根据用户和所有服务器，输出要选择的服务器
        """
        server_seq = torch.cat((static_server_seq, tmp_server_capacity, server_active), dim=-1)
        server_encoder_outputs = self.server_encoder(server_seq, graph)

        # get a pointer distribution over the encoder outputs using attention
        # (batch_size, server_len)
        probs = self.pointer(user, server_encoder_outputs, mask)
        # (batch_size, server_len)

        if self.policy == 'sample':
            # (batch_size, 1)
            idx = torch.multinomial(probs, num_samples=1)
            prob = torch.gather(probs, dim=1, index=idx)
        elif self.policy == 'greedy':
            prob, idx = torch.topk(probs, k=1, dim=-1)
        else:
            raise NotImplementedError

        prob = prob.squeeze(1)
        idx = idx.squeeze(1)

        return prob, idx

    def forward(self, env: DynamicEUAEnv, user_input_seq_with_stay, server_input_seq, masks, graph=None,
                has_user_num_per_sec=None):
        # 记录所有用户分配所用的时间，不包括计算指标的时间
        all_time = 0
        user_input_seq = user_input_seq_with_stay[:, :, :-1]
        batch_size = user_input_seq.size(0)
        server_len = server_input_seq.size(1)
        if has_user_num_per_sec is None:
            user_num_per_sec = env.get_user_num_per_sec_list()
        else:
            user_num_per_sec = has_user_num_per_sec

        # 记录真实分配情况
        # 服务器分配矩阵，加一是为了给index为-1的来赋值
        server_allocate_mat = torch.zeros(batch_size, server_len + 1, dtype=torch.long, device=self.device)

        # 服务器信息由三部分组成
        static_server_seq = server_input_seq[:, :, :2]
        tmp_server_capacity = server_input_seq[:, :, 2:].clone()

        user_encoder_outputs = self.user_encoder(user_input_seq)

        action_probs = []
        action_idx = []

        # 表示当前用户的索引
        i = -1
        indicators = []

        # 以总秒数作为循环
        for current_time in range(self.total_sec):

            current_user_len = user_num_per_sec[current_time]
            if current_user_len == 0:
                continue

            # 记录当前时间
            start_time = time.time()

            # 在这一秒要分配的用户
            for user_i in range(current_user_len):
                i += 1
                mask = masks[:, i]
                user_code = user_encoder_outputs[:, i, :].unsqueeze(1)
                # 根据当前用户和所有服务器，输出要选择的服务器
                _, _ = self.choose_server_id(mask, user_code, static_server_seq, tmp_server_capacity,
                                                  server_allocate_mat[:, :-1].unsqueeze(-1) > 0, graph)

            # 记录时间
            all_time += time.time() - start_time

        return all_time
