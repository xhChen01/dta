import pickle
import json
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from rdkit import Chem

# 加载数据集中亲和度信息
def load_data(dataset, max_smi_len=0, max_seq_len=0, seed=42, is_load_1d_data = False):
    
    data = {}
    data['path'] = 'dataset/' + dataset + '/'
    
    # 加载可能包含特殊字符或用旧版本 Python 生成的 pickle 文件时，推荐使用 encoding='latin1' 参数。
    affinity = pickle.load(open(data['path'] + 'affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        # pkd = -log10(Kd)   pki = -log10(Ki)，将数值范围压缩到合理范围使其更符合正太分布，便于模型训练
        affinity = -np.log10(affinity / 1e9)

    X_train, X_test, y_train, y_test = split_train_test_dataset(affinity, seed)
    n_drugs, n_targets = affinity.shape

    data['X_train'] = X_train
    data['X_test'] = X_test
    data['y_train'] = y_train
    data['y_test'] = y_test
    data['n_drugs'] = n_drugs
    data['n_targets'] = n_targets

    if(is_load_1d_data):
        load_1d_data(data, max_smi_len, max_seq_len)

    return data

# 划分数据集，训练集和测试集5:1
def split_train_test_dataset(affinity, seed):
    # 找到所有不包含NaN值的索引
    rows, cols = np.where(np.isnan(affinity) == False)
    X = np.array(list(zip(rows, cols)))
    y = np.array(affinity[rows, cols])
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1667, random_state=seed)

    return X_train, X_test, y_train, y_test

def load_1d_data(data, max_smi_len, max_seq_len):

    X_train, X_test = data['X_train'], data['X_test']
    ligands =  json.load(open(data['path']+"drugs.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(data['path']+"targets.txt"), object_pairs_hook=OrderedDict)

    ligands_names = list(ligands.keys())
    proteins_names = list(proteins.keys())
    ligands_features = []
    proteins_features = []

    # 加载药物和蛋白质序列特征向量
    for ligand_name in ligands_names:
        # (n_ligands, max_smi_len)
        ligands_features.append(label_smiles(ligands[ligand_name], max_smi_len, CHARCANSMISET))
    for protein_name in proteins_names:
        # (n_proteins, max_seq_len)
        proteins_features.append(label_sequence(proteins[protein_name], max_seq_len, CHARPROTSET))
    
    # 转换为 numpy 数组
    ligands_features = np.array(ligands_features)
    proteins_features = np.array(proteins_features)
    
    data['max_smi_len'] = max_smi_len
    data['max_seq_len'] = max_seq_len
    data['charsmiset_size'] = CHARCANSMILEN
    data['charprotset_size'] = CHARPROTLEN
    data['ligands_features'] = ligands_features
    data['proteins_features'] = proteins_features


CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
    "U": 19, "T": 20, "W": 21, 
    "V": 22, "Y": 23, "X": 24, 
    "Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
    ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
    "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
    "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
    "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
    "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
    "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
    "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
    "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
    "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
    "t": 61, "y": 62}
CHARCANSMILEN = 62


def label_smiles(line, max_smi_len, smi_ch_ind):
	X = np.zeros(max_smi_len)
	for i, ch in enumerate(line[:max_smi_len]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_sequence(line, max_seq_len, smi_ch_ind):
	X = np.zeros(max_seq_len)

	for i, ch in enumerate(line[:max_seq_len]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

# 加载药物分子图
def get_drug_molecule_graph(ligands):
    # 创建有序字典存储药物分子图
    smile_graph = OrderedDict()
    for d in ligands.keys():
        # 获取当前药物的SMILES字符串
        # 先将SMILES转换为Mol对象，再转换回SMILES字符串，确保SMILES格式标准化
        # isomericSmiles=True保留异构体信息（如立体化学）
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        # 将标准化后的SMILES字符串转换为分子图表示
        smile_graph[d] = smile_to_graph(lg)

    return smile_graph

def smile_to_graph(ligand):
    # 将SMILES字符串转换为RDKit的Mol对象
    mol = Chem.MolFromSmiles(ligand)
    # 获取分子中的原子数量
    atom_num = mol.GetNumAtoms()

    # 提取每个原子的特征
    features = []
    for atom in mol.GetAtoms():
        # 使用atom_features函数提取原子特征
        feature = atom_features(atom)
        # 对特征进行归一化处理，确保特征值在合理范围内
        features.append(feature / sum(feature))


   # 提取分子中的化学键信息
    edges = []
    # 忽略了GetBondType（边的类型，如单键、双键、三键等），只考虑了化学键的方向（从开始原子指向结束原子）
    for bond in mol.GetBonds():
        # 获取每个化学键的起始原子和结束原子索引
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    edge_index = []
    # 初始化邻接矩阵
    mol_adj = np.zeros((atom_num, atom_num))
    # 填充邻接矩阵，有边的位置设为1
    for e1, e2 in edges:
        mol_adj[e1, e2] = 1
    # 添加自环（对角线元素设为1）
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    # 提取邻接矩阵中值大于等于0.5的元素的索引（即所有边的位置）
    index_row, index_col = np.where(mol_adj >= 0.5)
    # 构建边索引列表
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    return atom_num, features, edge_index

def atom_features(atom):

    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))