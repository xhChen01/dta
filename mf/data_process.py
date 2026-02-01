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
def load_data(dataset, seed=42):
    # 加载可能包含特殊字符或用旧版本 Python 生成的 pickle 文件时，推荐使用 encoding='latin1' 参数。
    affinity = pickle.load(open('dataset/' + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        # pkd = -log10(Kd)   pki = -log10(Ki)，将数值范围压缩到合理范围使其更符合正太分布，便于模型训练
        affinity = -np.log10(affinity / 1e9)

    X_train, X_test, y_train, y_test = split_train_test_dataset(affinity, seed)
    n_drugs, n_targets = affinity.shape

    return n_drugs, n_targets, X_train, X_test, y_train, y_test

# 划分数据集，训练集和测试集5:1
def split_train_test_dataset(affinity, seed):
    # 找到所有不包含NaN值的索引
    rows, cols = np.where(np.isnan(affinity) == False)
    X = np.array(list(zip(rows, cols)))
    y = np.array(affinity[rows, cols])
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1667, random_state=seed)

    return X_train, X_test, y_train, y_test

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