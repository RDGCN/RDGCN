import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
polarity_set = ['positive', 'negative', 'neutral']
data_name = 'Laptop'
K_path = None
seed = 10

bert = False
lowercase = True
pad_unk_set = ['<pad>', '<unk>']
max_tree_dis = 10
glove_prefix = './glove/'
glove_path = './glove/glove.840B.300d.txt'
glove_dim = 300
max_sequence_len = None

pretrained_bert_name = './bert/bert-base-uncased'
bert_dim = 768
bert_dropout = 0.3
bert_lr = 2e-5
bert_adam_epsilon = 1e-8

batch_size = 32
learning_rate = 0.002
weight_decay = 0.0
epoch_num = 20

pos_embed_dim = 30
dep_embed_dim = 30
loc_embed_dim = 30
input_dropout = 0.7

rnn_layer_num = 2
rnn_hidden_dim = 50
rnn_dropout = 0.1
rnn_bidirection = True

gnn_layer_num = 2
gnn_output_dim = 50
gnn_dropout = 0.1
gnn_type = 'GCN'

RL_K = 0.1
RL_K_min = 0.1
RL_K_max = 2
RL_K_log = []
RL_S = 0.1
RL_b = 2
RL_R = 10
RL_memory = []
