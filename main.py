import os
import copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import toolbox.para_loader as pl
from sklearn import metrics
from model.RDGNN import RDGNN
from model.RDGNN_BERT import RDGNN_BERT
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from toolbox.vocab_generator import tokenizer_generation, embedding_generation, DataSet, Tokenizer4BertGCN, DataSet_BERT
from toolbox.data_preprocessor import syn_adj_generation
from pprint import pprint

def seed_setting(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class RunModel(object):
    def __init__(self, K):
        self.file_prefix = './data/' + pl.data_name
        if pl.bert:
            pl.max_sequence_len = 85
            self.tokenizer = Tokenizer4BertGCN(pl.max_sequence_len, pl.pretrained_bert_name)
            self.bert = BertModel.from_pretrained(pl.pretrained_bert_name)
            vocab_dep = pickle.load(open(self.file_prefix + '/tokenizer.dat', 'rb')).vocab_dep
            dep_vocab_size = len(vocab_dep)
            self.model = RDGNN_BERT(self.bert, dep_vocab_size, K).to(pl.device)
            [train_set, test_set] = [DataSet_BERT(self.file_prefix, file_type, self.tokenizer, vocab_dep, pl.max_sequence_len)
                                     for file_type in ['train', 'test']]
            self.train_dataloader = DataLoader(dataset=train_set, batch_size=pl.batch_size, shuffle=True)
            self.test_dataloader = DataLoader(dataset=test_set, batch_size=pl.batch_size)
            self.comp_outline = list(train_set[0].keys())
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = self.get_bert_optimizer(self.model)
        else:
            self.tokenizer = tokenizer_generation(self.file_prefix)
            syn_adj_generation(self.file_prefix, self.tokenizer.vocab_dep)
            self.embedding = embedding_generation(self.file_prefix, self.tokenizer.vocab_tok)
            [train_set, test_set] = [DataSet(self.file_prefix, file_type, self.tokenizer)
                                     for file_type in ['train', 'test']]
            self.train_dataloader = DataLoader(dataset=train_set, batch_size=pl.batch_size, shuffle=True)
            self.test_dataloader = DataLoader(dataset=test_set, batch_size=pl.batch_size)
            self.comp_outline = list(train_set[0].keys())
            self.criterion = nn.CrossEntropyLoss()
            self.model = RDGNN(len(self.tokenizer.vocab_pos), len(self.tokenizer.vocab_dep),
                               len(self.tokenizer.vocab_loc), self.embedding, K).to(pl.device)
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=pl.learning_rate, weight_decay=pl.weight_decay)
    def get_bert_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': pl.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=pl.bert_lr, eps=pl.bert_adam_epsilon)
        return optimizer
    def train(self, search_flag):
        max_eval_acc = 0
        max_eval_f1 = 0
        batch_num = 0
        for epoch_idx in range(pl.epoch_num):
            for batch_idx, batch_data in enumerate(self.train_dataloader):
                batch_num += 1
                self.model.train()
                self.optimizer.zero_grad()
                input = [batch_data[comp].to(pl.device) for comp in self.comp_outline]
                output = self.model(input, search_flag)
                target = batch_data['polarity'].to(pl.device)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_num % pl.RL_b == 0:
                    eval_acc, eval_f1 = self.evaluate()
                    if search_flag:
                        self.model.rnn_gnn.reward_and_punishment(eval_acc)
                    if eval_acc > max_eval_acc:
                        max_eval_acc = eval_acc
                        self.best_model = copy.deepcopy(self.model)
                    if eval_f1 > max_eval_f1:
                        max_eval_f1 = eval_f1
                    # print('loss: {:.4f}, eval_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), eval_acc, eval_f1))
        if not search_flag:
            print('max_eval_acc: {:.4f}, max_eval_f1: {:.4f}'.format(max_eval_acc, max_eval_f1))
    def evaluate(self):
        self.model.eval()
        eval_correct_num, eval_total_num = 0, 0
        target_all, output_all = None, None
        with torch.no_grad():
            for batch_data in self.test_dataloader:
                input = [batch_data[comp].to(pl.device) for comp in self.comp_outline]
                target = batch_data['polarity'].to(pl.device)
                output = self.model(input, False)
                eval_correct_num += (torch.argmax(output, -1) == target).sum().item()
                eval_total_num += len(output)
                target_all = torch.cat((target_all, target), dim=0) if target_all is not None else target
                output_all = torch.cat((output_all, output), dim=0) if output_all is not None else output
        eval_acc = eval_correct_num / eval_total_num
        f1 = metrics.f1_score(target_all.cpu(), torch.argmax(output_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return eval_acc, f1
    def test(self):
        self.model = self.best_model
        self.model.eval()
        test_acc, test_f1 = self.evaluate()
        print('max_test_acc: {:.4f}, max_test_f1: {:.4f}'.format(test_acc, test_f1))

if __name__ == '__main__':
    # seed_setting(pl.seed)
    pl.K_path = './'+pl.data_name+'_K'
    if not os.path.exists(pl.K_path):
        run_model_ = RunModel(0.)
        run_model_.train(True)
        K_info = {'K':None, 'R':None}
        K_info['K'] = run_model_.model.rnn_gnn.RL_K
        K_info['RL_memory'] = run_model_.model.rnn_gnn.RL_memory
        pickle.dump(K_info, open(pl.K_path, 'wb'))
    pl.RL_b = 1
    K_info = pickle.load(open(pl.K_path, 'rb'))
    K = K_info['K']
    run_model = RunModel(K)
    run_model.train(False)
    run_model.test()

