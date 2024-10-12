import numpy as np
import pandas as pd
import torch
from networkx import Graph
from torch_geometric.data import Data
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from collections import defaultdict
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondStereo
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from model_params import smarts_file_path
from multiprocessing import Pool
import multiprocessing
import tqdm
import os


# 获取原子类型，如果原子的原子类型不在match_symbol_list里，则使用"other"覆盖
def get_symbol(mol):
    atoms = mol.GetAtoms()
    symbols = [atom.GetSymbol() for atom in atoms]
    match_symbol_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
    for i in range(len(symbols)):
        if symbols[i] not in match_symbol_list:
            symbols[i] = 'other'
    return symbols


# 获取杂化状态，如果原子的杂化状态不在match_hybridization_list里，则使用"other"覆盖
def get_hybridization(mol):
    atoms = mol.GetAtoms()
    hybridization = [atom.GetHybridization() for atom in atoms]
    match_hybridization_list = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
                                HybridizationType.SP3D, HybridizationType.SP3D2]
    for i in range(len(hybridization)):
        if hybridization[i] not in match_hybridization_list:
            hybridization[i] = 'other'
    return hybridization


# 获取手性类型（R或S），smiles中给出了分子中手性原子的RS构型则返回其构型（“R”或“S”），如果没有给出其RS构型信息，不论是否是手性原子，均返回“N”
def get_chirality_type(atom):
    try:
        ChiType = atom.GetProp('_CIPCode')
    except KeyError:
        ChiType = 'N'
    return ChiType


# 判断原子是否是电离中心
def is_ionization_center(atom, mol, smarts_df):
    atom_index = atom.GetIdx()  # 获取原子的序号
    for i in range(smarts_df.shape[0]):
        smarts = smarts_df.loc[i, 'SMARTS']
        center_index = smarts_df.loc[i, 'new_index']
        # 一个smarts中可能有一个或多个电离中心，当只有一个电离中心时，使用一个数字表示，当有多个电离中心时，使用多个数字表示，数字之间使用英文逗号分隔
        if ',' not in center_index:  # 不存在英文逗号，说明只有一个电离中心
            center_index = [int(center_index)]  # 从csv中读入的时候是str变量，需要转化为int，为了和下面统一，再将其转化为列表
        else:  # 否则存在两个或以上的电离中心
            center_index = center_index.split(",")  # 使用逗号分隔开
            center_index = [int(j) for j in center_index]  # 转化为int，使用列表表示
        patt = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(patt):  # 如果结构匹配上
            mapping = mol.GetSubstructMatches(patt)  # 这里的mapping是匹配上的结构在mol对象中的索引
            for idx in center_index:  # 这里的center_index是电离位点在smarts中的索引列表，idx则是其中每一个电离位点在smarts上的索引
                # mapping可以视为一个第一维为1的二维元组，即，有两层括号的二维元组，如：((4, 2, 1, 0, 5),)
                # 其中，每一个数字是分子与smarts一一匹配后，按照smarts中的原子顺序，给出的各个原子在分子中的索引
                # 例如，smiles为'N=1(N=C(C)N(C=1C)N)'，smarts为'[nX3:0]:[c:1]:[nX2:2]:[nX2:3]:[c:4]'，匹配结果为((4, 2, 1, 0, 5),)
                # 说明smarts中的0号原子子是smiles中的4号原子，smarts中的1号原子是smiles中的2号原子，smarts中的2号原子是smiles中的1号原子，
                # smarts中的3号原子是smiles中的0号原子，smarts中的4号原子是smiles中的5号原子
                # 而idx是smarts中电离中心的索引，所以mapping[0][idx]则是该电离中心在smiles中的索引，通过与atom_index判定相等则可以判断是否是电离中心
                if atom_index == mapping[0][idx]:
                    return True
    return False


# 判断所有分子的原子是否是电离中心，每一个批次的原子放在一个列表中
def is_ionization_center_all(data, smarts_df, acid_or_base, save_path, verbose=False):
    if acid_or_base == 'acid':
        a_or_b = 'A'
    elif acid_or_base == 'base':
        a_or_b = 'B'
    else:
        print('Error!')
        return 0
    smarts_df = smarts_df[smarts_df['Acid_or_base'] == a_or_b].reset_index(drop=True)
    ionization_center_bool_list = []
    for index_i, data_i in enumerate(data):
        if verbose:
            print('Processing batch ', index_i + 1, ' ....', sep='')
        # 该循环内是一个批次的数据
        ionization_center_bool_list_i = []  # 用于存储一个批次的数据
        for index_j in range(data_i.y.shape[0]):
            # 该循环内是一个分子的数据
            mol = Chem.MolFromSmiles(data_i[index_j].Smiles)
            mol_H = Chem.AddHs(mol)
            for index_k, atom in enumerate(mol.GetAtoms()):
                # 该循环内是一个原子的数据
                ionization_center_bool_list_i.append(is_ionization_center(atom, mol_H, smarts_df))
        ionization_center_bool_list_i = torch.tensor(ionization_center_bool_list_i)
        ionization_center_bool_list.append(ionization_center_bool_list_i)
    if save_path != None:
        torch.save(ionization_center_bool_list, save_path)
    return ionization_center_bool_list


# 导入smarts文件
smarts_df = pd.read_csv(smarts_file_path, delimiter='\t')


# 获取分子的原子特征，返回值为字典，键为特征名称，值为该名称下该分子内所有原子的特征，顺序与smiles中原子的出现顺序保持一致
def get_atoms_feature(mol):
    atoms = mol.GetAtoms()
    atoms_feature = defaultdict(list)
    atoms_feature['Symbol'] = get_symbol(mol)
    atoms_feature['Degree'] = [atom.GetDegree() for atom in atoms]
    atoms_feature['Formal_Charge'] = [atom.GetFormalCharge() for atom in atoms]
    atoms_feature['Radical_Electrons'] = [atom.GetNumRadicalElectrons() for atom in atoms]
    atoms_feature['Hybridization'] = get_hybridization(mol)
    atoms_feature['Aromaticity'] = [atom.GetIsAromatic() for atom in atoms]
    atoms_feature['Hydrogens'] = [atom.GetTotalNumHs() for atom in atoms]
    # 分子中只要有手性原子，不论smiles中是否给出了该原子的RS构型，则该原子的atom_feature['IsChirality']为1，否则为0
    atoms_feature['IsChirality'] = [atom.HasProp('_ChiralityPossible') for atom in atoms]
    # smiles中给出了分子中手性原子的RS构型则返回其构型（“R”或“S”），如果没有给出其RS构型信息，不论是否是手性原子，均返回“N”
    atoms_feature['Chirality_Type'] = [get_chirality_type(atom) for atom in atoms]
    atoms_feature['IsIonizationCenter'] = [is_ionization_center(atom, mol, smarts_df) for atom in atoms]
    return atoms_feature


# 匹配键特征和键的连接关系
def get_bonds_matched(mol):
    bonds = mol.GetBonds()
    bonds_connected_directed = get_bonds_connected(mol).T
    bonds_list = []
    for bond_connected in bonds_connected_directed:  # 遍历每一个连接关系
        for bond in bonds:  # 遍历每一条边
            # 如果边的起始原子索引和终止原子索引分别与边的连接关系的第一二个数字一一对应，或者与边的连接关系的第二、第一个数字一一对应，则添加到边列表
            if (bond.GetBeginAtomIdx() == bond_connected[0] and bond.GetEndAtomIdx() == bond_connected[1]) or (
                    bond.GetBeginAtomIdx() == bond_connected[1] and bond.GetEndAtomIdx() == bond_connected[0]):
                bonds_list.append(bond)
    return bonds_list


# 获取分子的键特征，返回值为字典，键为特征名称，值为该名称下该分子内所有键的特征，顺序与smiles中原子的出现顺序保持一致
def get_bonds_feature(mol):
    bonds = get_bonds_matched(mol)
    bonds_featrue = defaultdict(list)
    bonds_featrue['Bond_Type'] = [bond.GetBondType() for bond in bonds]
    bonds_featrue['Conjugation'] = [bond.GetIsConjugated() for bond in bonds]
    # bonds_featrue['Conjugation'] = [any(bond.GetIsConjugated() for bond in bonds)]
    bonds_featrue['Ring'] = [bond.IsInRing() for bond in bonds]
    # bonds_featrue['Ring'] = [any(bond.IsInRing() for bond in bonds)]
    bonds_featrue['Sterro'] = [bond.GetStereo() for bond in bonds]  # Z/E构型
    return bonds_featrue


# 获取边的连接情况
def get_bonds_connected(mol):
    bond_info = mol.GetBonds()
    bonds_connected = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bond_info]
    G = Graph(bonds_connected).to_directed()
    bonds_connected_directed = np.array(G.edges).T
    return bonds_connected_directed


# 对分子的原子特征和键特征进行独热编码，feature为一个分子中的全部原子或全部键的真实特征， match_feature为用于匹配的特征，feature_names_list为特征名列表
def get_molecular_one_hot_code(feature, match_feature, feature_names_list):
    one_hot_code = defaultdict(list)
    for feature_name in feature_names_list:
        one_hot_code[feature_name] = []
        for i in feature[feature_name]:
            one_hot_code[feature_name].append([int(i == a) for a in match_feature[feature_name]])
    return one_hot_code


# 使输入变为n行m列的列表（n表示分子的原子个数，m表示特征个数），要求输入为字典，每一个键为特征名称，值为该名称下该分子内所有原子的特征，顺序与smiles中原子的出现顺序保持一致
def get_not_normalized_feature_matrix(feature, feature_names_list):
    not_normalized_feature_matrix = []
    for i in range(len(feature[feature_names_list[0]])):
        not_normalized_feature_matrix_i = []
        for feature_name in feature_names_list:
            not_normalized_feature_matrix_i += feature[feature_name][i]
        not_normalized_feature_matrix.append(not_normalized_feature_matrix_i)
    return not_normalized_feature_matrix


# 归一化，按行（原子）归一化，不按列（特征）归一化是因为各特征的和可能为0，从而出现0做分母的情况
def get_normalized_feature_matrix(feature):
    sum = 0
    for i in range(len(feature)):
        for j in range(len(feature[0])):
            sum += feature[i][j]
        for j in range(len(feature[0])):
            feature[i][j] /= sum
        sum = 0
    return feature


# 处理单个分子数据，返回归一化后的原子特征独热编码矩阵，归一化后的键特征独热编码矩阵（行为原子或键，列为特征），键的连接情况和分子的原子数
def process_molecular_data(smiles):
    mol = MolFromSmiles(smiles)
    # 获取原子特征和键特征
    atoms_feature = get_atoms_feature(mol)
    bonds_feature = get_bonds_feature(mol)
    # 定义用于匹配的原子特征
    match_atoms_feature = defaultdict(list)
    match_atoms_feature['Symbol'] = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At',
                                     'other']
    match_atoms_feature['Degree'] = [0, 1, 2, 3, 4, 5]
    match_atoms_feature['Formal_Charge'] = [-1, 0, 1]
    match_atoms_feature['Radical_Electrons'] = [0, 1, 2, 3, 4]
    match_atoms_feature['Hybridization'] = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
                                            HybridizationType.SP3D, HybridizationType.SP3D2, 'other']
    match_atoms_feature['Aromaticity'] = [0, 1]
    match_atoms_feature['Hydrogens'] = [0, 1, 2, 3, 4]
    match_atoms_feature['IsChirality'] = [0, 1]
    match_atoms_feature['Chirality_Type'] = ['R', 'S']
    match_atoms_feature['IsIonizationCenter'] = [False, True]
    # 定义用于匹配的键特征
    match_bonds_feature = defaultdict(list)
    match_bonds_feature['Bond_Type'] = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
    match_bonds_feature['Conjugation'] = [False, True]
    match_bonds_feature['Ring'] = [False, True]
    match_bonds_feature['Sterro'] = [BondStereo.STEREONONE, BondStereo.STEREOANY, BondStereo.STEREOZ,
                                     BondStereo.STEREOE]
    # 定义特征名列表
    atoms_feature_names_list = list(match_atoms_feature.keys())
    bonds_feature_names_list = list(match_bonds_feature.keys())
    # 独热编码
    atoms_one_hot_code = get_molecular_one_hot_code(atoms_feature, match_atoms_feature, atoms_feature_names_list)
    bonds_one_hot_code = get_molecular_one_hot_code(bonds_feature, match_bonds_feature, bonds_feature_names_list)
    # 整理独热编码的结果，使结果变为n行m列的列表（n表示分子的原子个数，m表示特征个数）
    atoms_one_hot_code = get_not_normalized_feature_matrix(atoms_one_hot_code, atoms_feature_names_list)
    bonds_one_hot_code = get_not_normalized_feature_matrix(bonds_one_hot_code, bonds_feature_names_list)
    # 归一化
    atoms_one_hot_code = get_normalized_feature_matrix(atoms_one_hot_code)
    bonds_one_hot_code = get_normalized_feature_matrix(bonds_one_hot_code)
    # 获取边连接情况
    bonds_connected = get_bonds_connected(mol)
    return atoms_one_hot_code, bonds_one_hot_code, bonds_connected, mol.GetNumAtoms()


# 获取用于构建图的数据
def get_graph_data(file_path):
    # 读取数据
    data = pd.read_csv(file_path)
    Smiles_list = data['Smiles']
    basicOrAcid = data['basicOrAcidic']
    y = data['pKa']
    print('Smiles encoding....')

    # 创建一个进程池
    num_cores = os.cpu_count()  # 获取计算机总线程数
    with Pool(processes=num_cores) as p:
        result = []
        with tqdm.tqdm(total=len(Smiles_list)) as pbar:
            for i, processed_data in enumerate(p.imap(process_molecular_data, Smiles_list)):
                result.append(processed_data)
                pbar.update()
        x = result

    # graph_data是一个长度为4的列表
    # 第一个元素是一个由Smiles构成的列表，长度为NumMolecules
    # 第二个元素是长度为1的列表，有四层括号：
    #   1.最外层（第一层）是一个由特征构成的列表，包含了所有分子的特征信息，该列表长度为NumMolecules
    #   2.次外层（第二层）每一个元素表示一个分子的所有原子特征信息，是一个长度为4的元组，其中（即第三层）：
    #       （1）第一个元素是一个二维列表，形状为(NumAtoms,NumAtomsFeatures)（即第三、四层）
    #       （2）第二个元素是一个二维列表，形状为(NumBonds,NumBondsFeatures)（即第三、四层）
    #       （3）第三个元素是一个二维<class 'networkx.classes.reportviews.OutEdgeView'>对象，是一个可迭代数据类型，
    #       形状为(NumBonds,2)，表示键的连接情况，第一列是键的起点节点，第二列是终点节点（即第三、四层）
    #       （4）第四个元素是一个int类型的数字，表示该分子中的原子数
    # 第三个元素是一个由“acidic”或“basic”字符串构成的列表，表示该电离基团是酸性基团还是碱性基团，长度为NumMolecules
    # 第四个元素是一个由pKa值构成的列表，长度为NumMolecules
    graph_data = [Smiles_list.tolist(), x] + [basicOrAcid.tolist()] + [y.tolist()]
    return graph_data


# 从dataframe中获取数据
def get_graph_data_from_df(data):
    # 读取数据
    Smiles_list = data['Smiles']
    basicOrAcid = data['basicOrAcidic']
    y = data['pKa']
    print('Smiles encoding....')

    # 创建一个进程池
    num_cores = os.cpu_count()  # 获取计算机总线程数
    with Pool(processes=num_cores) as p:
        result = []
        with tqdm.tqdm(total=len(Smiles_list)) as pbar:
            for i, processed_data in enumerate(p.imap(process_molecular_data, Smiles_list)):
                result.append(processed_data)
                pbar.update()
        x = result

    # graph_data是一个长度为4的列表
    # 第一个元素是一个由Smiles构成的列表，长度为NumMolecules
    # 第二个元素是长度为1的列表，有四层括号：
    #   1.最外层（第一层）是一个由特征构成的列表，包含了所有分子的特征信息，该列表长度为NumMolecules
    #   2.次外层（第二层）每一个元素表示一个分子的所有原子特征信息，是一个长度为4的元组，其中（即第三层）：
    #       （1）第一个元素是一个二维列表，形状为(NumAtoms,NumAtomsFeatures)（即第三、四层）
    #       （2）第二个元素是一个二维列表，形状为(NumBonds,NumBondsFeatures)（即第三、四层）
    #       （3）第三个元素是一个二维<class 'networkx.classes.reportviews.OutEdgeView'>对象，是一个可迭代数据类型，
    #       形状为(NumBonds,2)，表示键的连接情况，第一列是键的起点节点，第二列是终点节点（即第三、四层）
    #       （4）第四个元素是一个int类型的数字，表示该分子中的原子数
    # 第三个元素是一个由“acidic”或“basic”字符串构成的列表，表示该电离基团是酸性基团还是碱性基团，长度为NumMolecules
    # 第四个元素是一个由pKa值构成的列表，长度为NumMolecules
    graph_data = [Smiles_list.tolist(), x] + [basicOrAcid.tolist()] + [y.tolist()]
    return graph_data


# 构建图，输入是一个csv文件
def get_graph(file_path):
    data = get_graph_data(file_path)
    graph_record = []
    MoleculesIdx = 0
    for Smiles, graph_info, basicOrAcidic, pKa in zip(*data):
        # 从每个分子的特征中获取原子特征，键特征，键的连接情况，原子数四类特征
        atoms_feature = graph_info[0]
        bonds_feature = graph_info[1]
        bonds_connected = graph_info[2]
        atoms_number = graph_info[3]
        # 使用Data类管理每个分子的图数据
        graph_molecule = Data(x=torch.asarray(np.array(atoms_feature), dtype=torch.float32),
                              edge_feature=torch.asarray(np.array(bonds_feature), dtype=torch.float32),
                              edge_index=torch.tensor(bonds_connected, dtype=torch.long),
                              y=torch.tensor(pKa, dtype=torch.float32),
                              Smiles=Smiles,
                              bisicOrAcidic=basicOrAcidic,
                              atoms_number=atoms_number,
                              index=MoleculesIdx)
        MoleculesIdx += 1
        graph_record.append(graph_molecule)
    imd = InMemoryDataset()
    _data, _slice = imd.collate(graph_record)
    return _data, _slice


# 构建图，输入是一个dataframe
def get_graph_from_df(data):
    data = get_graph_data_from_df(data)
    graph_record = []
    MoleculesIdx = 0
    for Smiles, graph_info, basicOrAcidic, pKa in zip(*data):
        # 从每个分子的特征中获取原子特征，键特征，键的连接情况，原子数四类特征
        atoms_feature = graph_info[0]
        bonds_feature = graph_info[1]
        bonds_connected = graph_info[2]
        atoms_number = graph_info[3]
        # 使用Data类管理每个分子的图数据
        graph_molecule = Data(x=torch.asarray(np.array(atoms_feature), dtype=torch.float32),
                              edge_feature=torch.asarray(np.array(bonds_feature), dtype=torch.float32),
                              edge_index=torch.tensor(bonds_connected, dtype=torch.long),
                              y=torch.tensor(pKa, dtype=torch.float32),
                              Smiles=Smiles,
                              bisicOrAcidic=basicOrAcidic,
                              atoms_number=atoms_number,
                              index=MoleculesIdx)
        MoleculesIdx += 1
        graph_record.append(graph_molecule)
    imd = InMemoryDataset()
    _data, _slice = imd.collate(graph_record)
    return _data, _slice


if __name__ == '__main__':
    multiprocessing.freeze_support()
