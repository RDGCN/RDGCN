import os
import json
import networkx as nx
import toolbox.para_loader as pl
from pprint import pprint

def syn_dep_adj_generation(head, dep, vocab_dep):
    syn_dep_edge = []
    for node_s_id, (node_e_id, d) in enumerate(zip(head, dep)):
        if node_e_id == 0:
            continue
        syn_dep_edge.append([node_s_id, node_e_id-1, vocab_dep.word_to_id(d)])
    return syn_dep_edge

def syn_dis_adj_generation(head):
    syn_nx = syn_network_generation(head)
    syn_dis_edge = []
    for node_s_id in syn_nx.nodes:
        for node_e_id in syn_nx.nodes:
            try:
                tree_distance = nx.dijkstra_path_length(syn_nx, source=node_s_id, target=node_e_id)
                tree_distance = tree_distance if tree_distance <= pl.max_tree_dis else pl.max_tree_dis
            except:
                tree_distance = pl.max_tree_dis
            syn_dis_edge.append([node_s_id, node_e_id, tree_distance])
    return syn_dis_edge

def syn_network_generation(head):
    syn_nx = nx.Graph()
    syn_nx.add_nodes_from(range(len(head)))
    syn_nx.add_edges_from([(node_1, node_2 - 1) for node_1, node_2 in enumerate(head) if node_2 != 0])
    return syn_nx

def syn_adj_generation(file_prefix, vocab_dep):
    for file_type in ['train', 'test']:
        if os.path.exists(file_prefix + '/' + file_type + '_preprocessed.json'):
            continue
        else:
            with open(file_prefix + '/' + file_type + '.json', 'r') as f:
                file = json.load(f)
                for data in file:
                    head = data['head']
                    dep = data['deprel']
                    data['syn_dep_adj'] = syn_dep_adj_generation(head, dep, vocab_dep)
                    data['syn_dis_adj'] = syn_dis_adj_generation(head)
                file_ = open(file_prefix + '/' + file_type + '_preprocessed.json', 'w')
                file_.write(json.dumps(file))
                file_.close()
                print(file_prefix + '/' + file_type + '_preprocessed.json has been saved...')
