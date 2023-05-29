
import torch
import torch.nn as nn
import torch.nn.functional as F
import toolbox.para_loader as pl


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(pl.bert_dim))
        self.b_2 = nn.Parameter(torch.zeros(pl.bert_dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class RDGNN_BERT(nn.Module):
    def __init__(self, bert, dep_vocab_size, K):
        super().__init__()
        self.bert_gnn = BERT_GNN(bert, dep_vocab_size, K)
        self.classifier = nn.Linear(pl.gnn_output_dim, len(pl.polarity_set))
    def forward(self, input, search_flag):
        output = self.bert_gnn(input, search_flag)
        result = self.classifier(output)
        return result

class BERT_GNN(nn.Module):
    def __init__(self, bert, dep_vocab_size, K):
        super().__init__()
        self.bert = bert
        self.max_tree_dis = pl.max_tree_dis
        self.gnn_input_dim = pl.bert_dim
        self.gnn_hidden_dim = pl.gnn_output_dim
        self.bert_drop = nn.Dropout(pl.bert_dropout)
        self.gnn_drop = nn.Dropout(pl.gnn_dropout)
        self.layernorm = LayerNorm()
        self.dep_embedding = nn.Embedding(dep_vocab_size, pl.dep_embed_dim)
        self.dep_imp_function = DEP_IMP(pl.dep_embed_dim)
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
        [bert_indices, bert_ids, bert_mask, src_mask, aspect_mask, syn_dep_adj, syn_dis_adj, _] = input
        self.batch_size = src_mask.size(0)
        overall_max_len = src_mask.size(1)
        bert_output, _ = self.bert(bert_indices, attention_mask=bert_mask, token_type_ids=bert_ids)
        bert_output = self.layernorm(bert_output)
        bert_output = self.bert_drop(bert_output)
        syn_dep_adj_ = self.dep_imp_function(self.dep_embedding.weight, syn_dep_adj, overall_max_len, self.batch_size)
        dep_adj = syn_dep_adj_.float()
        if search_flag:
            syn_dis_adj_ = self.dis_imp_function_search(syn_dis_adj)
        else:
            syn_dis_adj_ = self.dis_imp_function(syn_dis_adj)
        dis_adj = syn_dis_adj_.float()
        A = torch.add(dep_adj, dis_adj)
        gnn_input = self.bert_drop(bert_output)
        gnn_output = gnn_input
        for layer_id in range(self.gnn_layer_num):
            gnn_output = self.W[layer_id](torch.matmul(A, gnn_output))
            gnn_output = F.relu(gnn_output)
            gnn_output = self.gnn_drop(gnn_output)
        output = F.relu(self.transition(gnn_output))
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.gnn_hidden_dim)
        output = (output*aspect_mask).sum(dim=1) / asp_wn
        return output
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
