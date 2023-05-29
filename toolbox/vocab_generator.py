import os
import json
import numpy as np
import toolbox.para_loader as pl
import pickle
from collections import Counter
from torch.utils.data import Dataset
import networkx as nx
import warnings
from pprint import pprint
from transformers import BertTokenizer
warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=1000, threshold=np.inf)

def tokenizer_generation(file_prefix):
    file_path = file_prefix + '/tokenizer.dat'
    if os.path.exists(file_path):
        tokenizer = pickle.load(open(file_path, 'rb'))
    else:
        tokenizer = Tokenizer(file_prefix)
        pickle.dump(tokenizer, open(file_path, 'wb'))
    return tokenizer

class Vocab(object):
    def __init__(self, vocab, pad_unk_flag=True):
        self.vocab = vocab
        self.vocab_reverse = {word_id:word for word, word_id in vocab.items()}
        if pad_unk_flag:
            self.pad_word = pl.pad_unk_set[0]
            self.unk_word = pl.pad_unk_set[1]
            self.pad_id = vocab[self.pad_word]
            self.unk_id = vocab[self.unk_word]
    def word_to_id(self, word):
        return self.vocab.get(word, self.unk_id)
    def id_to_word(self, word_id):
        return self.vocab_reverse.get(word_id, self.unk_word)
    def has_word(self, word):
        return word in self.vocab
    def __len__(self):
        return len(self.vocab)

class Tokenizer(object):
    def __init__(self, file_prefix):
        self.vocab_tok, self.vocab_pos, self.vocab_dep, self.vocab_loc, self.vocab_cls = self.vocab_loading(file_prefix)
    def vocab_loading(self, file_prefix):
        vocab_tok_file = file_prefix + '/vocab_tok.vocab'
        vocab_pos_file = file_prefix + '/vocab_pos.vocab'
        vocab_dep_file = file_prefix + '/vocab_dep.vocab'
        vocab_loc_file = file_prefix + '/vocab_loc.vocab'
        vocab_cls_file = file_prefix + '/vocab_cls.vocab'
        train_tok_set, train_pos_set, train_dep_set, train_max_len = self.data_loading(file_prefix + '/train.json')
        test_tok_set, test_pos_set, test_dep_set, test_max_len = self.data_loading(file_prefix + '/test.json')
        self.vocab_storage(vocab_tok_file, train_tok_set + test_tok_set)
        self.vocab_storage(vocab_pos_file, train_pos_set + test_pos_set)
        self.vocab_storage(vocab_dep_file, train_dep_set + test_dep_set)
        self.max_sentence_len = max(train_max_len, test_max_len)
        vocab_loc = {str(loc): loc_id for loc_id, loc
                     in enumerate(pl.pad_unk_set + list(range(-self.max_sentence_len, self.max_sentence_len)))}
        self.vocab_storage(vocab_loc_file, vocab_loc)
        vocab_cls = {str(cls): cls_id for cls_id, cls in enumerate(pl.polarity_set)}
        self.vocab_storage(vocab_cls_file, vocab_cls)
        vocab_tok = pickle.load(open(vocab_tok_file, 'rb'))
        vocab_pos = pickle.load(open(vocab_pos_file, 'rb'))
        vocab_dep = pickle.load(open(vocab_dep_file, 'rb'))
        vocab_loc = pickle.load(open(vocab_loc_file, 'rb'))
        vocab_cls = pickle.load(open(vocab_cls_file, 'rb'))
        vocab_tok = Vocab(vocab_tok)
        vocab_pos = Vocab(vocab_pos)
        vocab_dep = Vocab(vocab_dep)
        vocab_loc = Vocab(vocab_loc)
        vocab_cls = Vocab(vocab_cls, False)
        return vocab_tok, vocab_pos, vocab_dep, vocab_loc, vocab_cls
    def data_loading(self, file_path):
        with open(file_path) as f:
            file = json.load(f)
            tok_set = []
            pos_set = []
            dep_set = []
            max_len = 0
            for data in file:
                token = data['token']
                pos = data['pos']
                dep = data['deprel']
                if pl.lowercase:
                    token = [w.lower() for w in token]
                    pos = [p.lower() for p in pos]
                    dep = [d.lower() for d in dep]
                tok_set.extend(token)
                pos_set.extend(pos)
                dep_set.extend(dep)
                max_len = max(len(data['token']), max_len)
        return tok_set, pos_set, dep_set, max_len
    def vocab_storage(self, file_path, vocab_set):
        if isinstance(vocab_set, list):
            word2freq = Counter(vocab_set)
            word2freq = word2freq.most_common()
            word2freq = [(pl.pad_unk_set[0], 1e10), (pl.pad_unk_set[1], 1e10)] + word2freq
            vocab = {str(word[0]): id for id, word in enumerate(word2freq)}
        else:
            vocab = vocab_set
        with open(file_path, "wb") as f:
            pickle.dump(vocab, f)
            print(file_path + ' has been saved...')
    def sequence_transition(self, sequence, vocab):
        if pl.lowercase:
            sequence = [w.lower() for w in sequence]
        sequence_ = [vocab.word_to_id(w) for w in sequence]
        return self.truncating_padding(sequence_, vocab.pad_id)
    def truncating_padding(self, sequence, pad_id):
        if pl.max_sequence_len is None:
            max_len = self.max_sentence_len
        else:
            max_len = pl.max_sequence_len
        if max_len <= len(sequence):
            sequence_ = np.asarray(sequence[:max_len])
        else:
            sequence_ = (np.zeros(max_len) + pad_id).astype('int')
            sequence_[:len(sequence)] = sequence
        return sequence_

def embedding_generation(file_prefix, vocab):
    file_path = file_prefix + '/' + str(pl.glove_dim) + 'd_embedding.dat'
    if os.path.exists(file_path):
        embedding = pickle.load(open(file_path, 'rb'))
    else:
        embedding = np.zeros((len(vocab), pl.glove_dim))
        glove = load_glove(vocab)
        for i in range(len(vocab)):
            embedding[i] = glove.get(vocab.id_to_word(i))
        pickle.dump(embedding, open(file_path, 'wb'))
        print(file_path + ' has been saved...')
    return embedding

def load_glove(vocab):
    with open(pl.glove_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        glove = dict()
        for line in f:
            word_and_vec = line.rstrip().split()
            word = ''.join((word_and_vec[:-pl.glove_dim]))
            vec = word_and_vec[-pl.glove_dim:]
            if vocab.has_word(word):
                glove[word] = np.asarray(vec, dtype='float')
        for word in vocab.vocab.keys():
            if word not in glove.keys():
                glove[word] = np.random.uniform(-0.25, 0.25, pl.glove_dim)
        return glove

class DataSet(Dataset):
    def __init__(self, file_prefix, file_type, tokenizer):
        self.tokenizer = tokenizer
        self.max_sentence_len = tokenizer.max_sentence_len
        self.max_tree_dis = pl.max_tree_dis
        self.data_set = []
        file_path = file_prefix + '/' + file_type + '_preprocessed.json'
        with open(file_path, 'r') as f:
            file = json.load(f)
            for data in file:
                sentence_len = len(data['token'])
                tok_sequence = self.tokenizer.sequence_transition(data['token'], self.tokenizer.vocab_tok)
                pos_sequence = self.tokenizer.sequence_transition(data['pos'], self.tokenizer.vocab_pos)
                # dep_sequence = self.tokenizer.sequence_transition(data['deprel'], self.tokenizer.vocab_dep)
                root_id = self.tokenizer.vocab_dep.word_to_id('root')
                syn_dep_adj = data['syn_dep_adj']
                syn_dep_adj = dep_adj_expansion(syn_dep_adj, self.max_sentence_len, root_id)
                syn_dis_adj = data['syn_dis_adj']
                syn_dis_adj = dis_adj_expansion(syn_dis_adj, self.max_sentence_len, 0, self.max_tree_dis)
                for asp in data['aspects']:
                    asp_s_loc = asp['from']
                    asp_e_loc = asp['to']
                    asp_range = range(asp['from'], asp['to'])
                    loc_sequence = [str(context_loc - asp_s_loc) for context_loc in range(asp_s_loc)] + \
                                   [str(0) for _ in range(asp_s_loc, asp_e_loc)] + \
                                   [str(context_loc - (asp_e_loc-1)) for context_loc in range(asp_e_loc, sentence_len)]
                    loc_sequence = self.tokenizer.sequence_transition(loc_sequence, self.tokenizer.vocab_loc)
                    mask_sequence = [1 if node_loc in asp_range else 0 for node_loc in range(sentence_len)]
                    mask_sequence = self.tokenizer.truncating_padding(mask_sequence, self.tokenizer.vocab_tok.pad_id)
                    polarity = self.tokenizer.vocab_cls.vocab[asp['polarity']]
                    self.data_set.append({
                        'tok': tok_sequence,
                        'pos': pos_sequence,
                        # 'dep': dep_sequence,
                        'loc': loc_sequence,
                        'mask': mask_sequence,
                        'syn_dep_adj': syn_dep_adj,
                        'syn_dis_adj': syn_dis_adj,
                        'sentence_len': sentence_len,
                        'polarity': polarity})
    def __getitem__(self, idx):
        return self.data_set[idx]
    def __len__(self):
        return len(self.data_set)

def dep_adj_expansion(syn_adj, max_len, weight):
    if pl.max_sequence_len is not None:
        max_len = pl.max_sequence_len
    for node_id in range(max_len):
        syn_adj.append([node_id, node_id, weight])
    syn_nx = nx.Graph()
    syn_nx.add_nodes_from(range(max_len))
    syn_nx.add_weighted_edges_from(syn_adj)
    syn_adj_ = nx.adjacency_matrix(syn_nx).A
    return syn_adj_

def dis_adj_expansion(syn_adj, max_len, weight, max_tree_dis):
    if pl.max_sequence_len is not None:
        max_len = pl.max_sequence_len
    syn_adj_ = []
    for node_s_id in range(max_len):
        for node_e_id in range(max_len):
            if node_s_id == node_e_id:
                syn_adj_.append([node_s_id, node_e_id, weight])
            else:
                syn_adj_.append([node_s_id, node_e_id, max_tree_dis])
    syn_nx_ = nx.Graph()
    syn_nx_.add_nodes_from(range(max_len))
    syn_nx_.add_weighted_edges_from(syn_adj_)
    syn_adj_ = nx.adjacency_matrix(syn_nx_).A
    syn_nx = nx.Graph()
    syn_nx.add_weighted_edges_from(syn_adj)
    syn_adj = nx.adjacency_matrix(syn_nx).A
    ins_len = len(syn_adj)
    syn_adj_[:ins_len, :ins_len] = syn_adj
    return syn_adj_


class Tokenizer4BertGCN:
    def __init__(self, max_sentence_len, pretrained_bert_name):
        self.max_sentence_len = max_sentence_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class DataSet_BERT(Dataset):
    def __init__(self, file_prefix, file_type, tokenizer, vocab_dep, max_sentence_len):
        self.tokenizer = tokenizer
        self.max_sentence_len = max_sentence_len
        self.max_tree_dis = pl.max_tree_dis
        self.data_set = []
        file_path = file_prefix + '/' + file_type + '_preprocessed.json'
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        with open(file_path, 'r') as f:
            file = json.load(f)
            for data in file:
                root_id = vocab_dep.word_to_id('root')
                syn_dep_adj = data['syn_dep_adj']
                syn_dep_adj = dep_adj_expansion(syn_dep_adj, self.max_sentence_len, root_id)
                syn_dis_adj = data['syn_dis_adj']
                syn_dis_adj = dis_adj_expansion(syn_dis_adj, self.max_sentence_len, 0, self.max_tree_dis)
                text_list = data['token']
                for asp in data['aspects']:
                    polarity = polarity_dict[asp['polarity']]
                    term_start = asp['from']
                    term_end = asp['to']
                    left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]
                    left_tokens, term_tokens, right_tokens = [], [], []
                    left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []
                    for ori_i, w in enumerate(left):
                        for t in tokenizer.tokenize(w):
                            left_tokens.append(t)
                            left_tok2ori_map.append(ori_i)
                    asp_start = len(left_tokens)
                    offset = len(left)
                    for ori_i, w in enumerate(term):
                        for t in tokenizer.tokenize(w):
                            term_tokens.append(t)
                            term_tok2ori_map.append(ori_i + offset)
                    asp_end = asp_start + len(term_tokens)
                    offset += len(term)
                    for ori_i, w in enumerate(right):
                        for t in tokenizer.tokenize(w):
                            right_tokens.append(t)
                            right_tok2ori_map.append(ori_i + offset)
                    while len(left_tokens) + len(right_tokens) > self.max_sentence_len - 2 * len(term_tokens) - 3:
                        if len(left_tokens) > len(right_tokens):
                            left_tokens.pop(0)
                            left_tok2ori_map.pop(0)
                        else:
                            right_tokens.pop()
                            right_tok2ori_map.pop()
                    bert_tokens = left_tokens + term_tokens + right_tokens
                    bert_indices = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(bert_tokens) + \
                                   [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + \
                                   [tokenizer.sep_token_id]
                    context_asp_len = len(bert_indices)
                    paddings = [0] * (self.max_sentence_len - context_asp_len)
                    context_len = len(bert_tokens)
                    bert_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
                    src_mask = [0] + [1] * context_len + [0] * (self.max_sentence_len - context_len - 1)
                    aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
                    aspect_mask = aspect_mask + (self.max_sentence_len - len(aspect_mask)) * [0]
                    bert_mask = [1] * context_asp_len + paddings
                    bert_indices += paddings
                    bert_indices = np.asarray(bert_indices, dtype='int64')
                    bert_ids = np.asarray(bert_ids, dtype='int64')
                    bert_mask = np.asarray(bert_mask, dtype='int64')
                    src_mask = np.asarray(src_mask, dtype='int64')
                    aspect_mask = np.asarray(aspect_mask, dtype='int64')
                    self.data_set.append({
                        'bert_indices': bert_indices,
                        'bert_ids': bert_ids,
                        'bert_mask': bert_mask,
                        # 'asp_start': asp_start,
                        # 'asp_end': asp_end,
                        'src_mask': src_mask,
                        'aspect_mask': aspect_mask,
                        'syn_dep_adj': syn_dep_adj,
                        'syn_dis_adj': syn_dis_adj,
                        'polarity': polarity})
    def __getitem__(self, idx):
        return self.data_set[idx]
    def __len__(self):
        return len(self.data_set)
