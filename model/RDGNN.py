import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import toolbox.para_loader as pl
from pprint import pprint
torch.set_printoptions(linewidth=1000, threshold=np.inf)

class RDGNN(nn.Module):
    def __init__(self, pos_vocab_size, dep_vocab_size, loc_vocab_size, embedding, K):
        super(RDGNN, self).__init__()
        self.gnn_output_dim = pl.gnn_output_dim
        self.rnn_gnn = RNN_GNN(pos_vocab_size, dep_vocab_size, loc_vocab_size, embedding, K)
        self.classifier = nn.Linear(pl.gnn_output_dim, len(pl.polarity_set))
    def forward(self, input, search_flag):
        [_, _, _, mask, _, _, _, _] = input
        output, max_len = self.rnn_gnn(input, search_flag)
        mask = mask[:, :max_len]
        asp_len = mask.sum(dim=1).unsqueeze(-1)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.gnn_output_dim)
        output = (output*mask).sum(dim=1)/asp_len
        result = self.classifier(output)
        return result

class RNN_GNN(nn.Module):
    def __init__(self, pos_vocab_size, dep_vocab_size, loc_vocab_size, embedding, K):
        super(RNN_GNN, self).__init__()
        self.max_tree_dis = pl.max_tree_dis
        self.device = pl.device
        self.batch_size = None
        self.input_drop = nn.Dropout(pl.input_dropout)
        self.rnn_drop = nn.Dropout(pl.rnn_dropout)
        self.gnn_drop = nn.Dropout(pl.gnn_dropout)
        self.tok_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float), freeze=True)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pl.pos_embed_dim)
        self.dep_embedding = nn.Embedding(dep_vocab_size, pl.dep_embed_dim)
        self.loc_embedding = nn.Embedding(loc_vocab_size, pl.loc_embed_dim)
        self.rnn_input_dim = pl.glove_dim + pl.pos_embed_dim + pl.loc_embed_dim
        self.rnn_hidden_dim = pl.rnn_hidden_dim
        self.rnn_total_layer_num = pl.rnn_layer_num*2 if pl.rnn_bidirection else pl.rnn_layer_num
        self.rnn = nn.LSTM(self.rnn_input_dim, pl.rnn_hidden_dim, pl.rnn_layer_num,
                           True, True, pl.rnn_dropout, pl.rnn_bidirection)
        self.dep_imp_function = DEP_IMP(pl.dep_embed_dim)
        self.gnn_input_dim = self.gnn_hidden_dim = pl.rnn_hidden_dim*2 if pl.rnn_bidirection else pl.rnn_hidden_dim
        self.gnn_layer_num = pl.gnn_layer_num
        if pl.gnn_type == 'GCN':
            self.W = nn.ModuleList()
            for layer_id in range(pl.gnn_layer_num):
                input_dim = self.gnn_input_dim if layer_id==0 else self.gnn_hidden_dim
                self.W.append(nn.Linear(input_dim, self.gnn_hidden_dim, True))
        self.transition = nn.Linear(self.gnn_hidden_dim, pl.gnn_output_dim)
        self.K = K
        self.RL_K = pl.RL_K
        self.RL_K_min = pl.RL_K_min
        self.RL_K_max = pl.RL_K_max
        self.RL_K_log = pl.RL_K_log
        self.RL_S = pl.RL_S
        self.RL_R = pl.RL_R
        self.RL_memory = pl.RL_memory
        self.last_acc = 0.
        self.RL_stop_flag = False
    def forward(self, input, search_flag):
        [tok, pos, loc, _, syn_dep_adj, syn_dis_adj, sentence_len, _] = input
        self.batch_size = len(tok)
        overall_max_len = tok.shape[1]
        max_len = max(sentence_len)
        tok_vec = self.tok_embedding(tok)
        pos_vec = self.pos_embedding(pos)
        loc_vec = self.loc_embedding(loc)
        input = [tok_vec] + [pos_vec] + [loc_vec]
        input = torch.cat(input, dim=2)
        rnn_input = self.input_drop(input)
        self.rnn.flatten_parameters()
        state_shape = (self.rnn_total_layer_num, self.batch_size, self.rnn_hidden_dim)
        H_0 = C_0 = Variable(torch.zeros(*state_shape), requires_grad=False).to(self.device)
        rnn_output = nn.utils.rnn.pack_padded_sequence(rnn_input, sentence_len.cpu(),
                                                       batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(rnn_output, (H_0, C_0))
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        syn_dep_adj_ = self.dep_imp_function(self.dep_embedding.weight, syn_dep_adj, overall_max_len, self.batch_size)
        dep_adj = syn_dep_adj_[:, :max_len, :max_len].float()
        if search_flag:
            syn_dis_adj_ = self.dis_imp_function_search(syn_dis_adj)
        else:
            syn_dis_adj_ = self.dis_imp_function(syn_dis_adj)
        dis_adj = syn_dis_adj_[:, :max_len, :max_len].float()
        A = torch.add(dep_adj, dis_adj)
        gnn_input = self.rnn_drop(rnn_output)
        gnn_output = gnn_input
        for layer_id in range(self.gnn_layer_num):
            gnn_output = self.W[layer_id](torch.matmul(A, gnn_output))
            gnn_output = F.relu(gnn_output)
            gnn_output = self.gnn_drop(gnn_output)
        output = F.relu(self.transition(gnn_output))
        return output, max_len
    def dis_imp_function(self, adj):
        adj_ = (1-torch.pow(adj/self.max_tree_dis, self.max_tree_dis))*torch.exp(-self.K*adj)
        return adj_
    def dis_imp_function_search(self, adj):
        adj_ = (1-torch.pow(adj/self.max_tree_dis, self.max_tree_dis))*torch.exp(-self.RL_K*adj)
        return adj_
    def reward_and_punishment(self, acc):
        if not self.RL_stop_flag:
            reward = -1 if acc <= self.last_acc else +1
            self.RL_memory.append(reward)
            self.last_acc = acc
            if len(self.RL_memory)>=self.RL_R and abs(sum(self.RL_memory[-self.RL_R:])) ==0 and len(self.RL_K_log)>100:
                self.RL_stop_flag = True
            self.RL_K = self.RL_K + self.RL_S if reward == +1 else self.RL_K - self.RL_S
            self.RL_K = self.RL_K_max if self.RL_K > self.RL_K_max else self.RL_K
            self.RL_K = self.RL_K_min if self.RL_K < self.RL_K_min else self.RL_K
            self.RL_K_log.append(self.RL_K)

class DEP_IMP(nn.Module):
    def __init__(self, att_dim):
        super(DEP_IMP, self).__init__()
        self.q = nn.Linear(att_dim, 1)
    def forward(self, input, syn_dep_adj, overall_max_len, batch_size):
        query = self.q(input).T
        att_adj = F.softmax(query, dim=-1)
        att_adj = att_adj.unsqueeze(0).repeat(batch_size, overall_max_len, 1)
        att_adj = torch.gather(att_adj, 2, syn_dep_adj)
        att_adj[syn_dep_adj == 0.] = 0.
        return att_adj
