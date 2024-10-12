from abc import ABC
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger, TautomerEnumerator
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from sklearn.model_selection import train_test_split
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import DataLoader
from graph_creating import get_graph, smarts_df
from model_params import *
import copy
from datetime import datetime
from multiprocessing import Pool, cpu_count


# 一、构建图前的数据处理
# 标准化smiles函数
def standardize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    smi = Chem.MolToSmiles(mol)
    return smi


# 标准化互变异构体
def canonicalize_tautomer(smi):
    mol = Chem.MolFromSmiles(smi)
    te = TautomerEnumerator()
    mol = te.Canonicalize(mol)
    smi = Chem.MolToSmiles(mol)
    return smi


# 中和电荷，返回中和电荷后的mol对象，对于含有金属离子的分子而言，其正负电荷可能是平衡的，即整个分子不带电荷，这种分子是不会被中和电荷的，如乙醇钠
def uncharge(mol):
    neutralier = Uncharger()
    mol = neutralier.uncharge(mol)
    return mol


# 判断是否是无机物，通过是否含有碳原子判断，含有碳原子则视为有机物，不含碳原子则视为无机物，有机物返回False，无机物返回True
def is_inorganic_substance(mol):
    carbon = Chem.MolFromSmarts('[#6]')  # '[#6]'表示碳原子
    if mol.GetSubstructMatches(carbon) != ():
        return False
    else:
        return True


# 判断是否是混合物，如果是混合物，返回True，否则返回False
def is_mixture(mol):
    smi = Chem.MolToSmiles(mol)
    if '.' in smi:  # smiles含有'.'的物质为混合物
        return True
    else:
        return False


# 判断是否含有金属离子（是否是盐）
def is_salt(mol):
    metal_ions_list = ['[Li', '[Be', '[Na', '[Mg', '[Al', '[K', '[Ca', '[Sc', '[Ti', '[V', '[Cr', '[Mn', '[Fe', '[Co',
                       '[Ni', '[Cu', '[Zn', '[Ga', '[Rb', '[Sr', '[Y', '[Zr', '[Nb', '[Mo', '[Tc', '[Ru', '[Rh', '[Pd',
                       '[Ag', '[Cd', '[In', '[Sn', '[Cs', '[Ba', '[La', '[Ce', '[Pr', '[Nd', '[Pm', '[Sm', '[Eu', '[Gd',
                       '[Tb', '[Dy', '[Ho', '[Er', '[Tm', '[Yb', '[Lu', '[Hf', '[Ta', '[W', '[Re', '[Os', '[Ir', '[Pt',
                       '[Au', '[Hg', '[Tl', '[Pb', '[Bi', '[Po', '[Fr', '[Ra', '[Ac', '[Th', '[Pa', '[U', '[Np', '[Pu',
                       '[Am', '[Cm', '[Bk', '[Cf', '[Es', '[Fm', '[Md', '[No', '[Lr', '[Rf', '[Db', '[Sg', '[Bh', '[Hs',
                       '[Mt', '[Ds', '[Rg', '[Cn', '[Nh', '[Fl', '[Mc', '[Lv']
    smi = Chem.MolToSmiles(mol)
    for metal_ion in metal_ions_list:
        if metal_ion in smi:  # 如果分子的smiles中含有上述金属离子列表中的离子则认为该分子含有金属离子
            return True
    return False


# 删除金属离子和金属原子，以乙醇钠（'CC[O-].[Na+]'）为例，其调用remove_metal后的结果是'CC[O-]'，金属离子或原子被去除，'.'也被去除
def remove_metal(mol):
    metal_list = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                  'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                  'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                  'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra',
                  'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
                  'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv']
    mol_rm = Chem.RWMol(mol)
    atoms = mol.GetAtoms()  # 获取分子的每个原子对象
    for atom in atoms:
        if atom.GetSymbol() in metal_list:  # 如果该原子在金属列表中，即该原子是金属离子或金属原子
            idx = atom.GetIdx()  # 获取该原子的原子索引
            mol_rm.RemoveAtom(idx)  # 移除该原子
    smi = Chem.MolToSmiles(mol_rm)
    mol = Chem.MolFromSmiles(smi)
    return mol


# 筛选分子量，分子量介于[mw_low,mw_high]之间的分子返回True，否则返回False
def is_molecular_weight_moderate(mol, mw_low, mw_high):
    mw = CalcExactMolWt(mol)
    if mw_low <= mw <= mw_high:
        return True
    else:
        return False


# 判断分子是否含有可识别位点
def is_recognizable_ionization_sites_contained(mol, smarts_file_path, acid_or_base):
    if acid_or_base == 'acid':
        a_or_b = 'A'
    elif acid_or_base == 'base':
        a_or_b = 'B'
    else:
        print('Error!')
        return 0
    smarts_list = pd.read_csv(smarts_file_path, delimiter='\t')
    smarts_list = smarts_list[smarts_list['Acid_or_base'] == a_or_b]['SMARTS']
    # smarts_list = pd.read_csv(smarts_file_path, delimiter='\t')['SMARTS']
    for smarts in smarts_list:
        patt = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(patt):
            return True
    return False

# 调用函数处理数据
def worker(args):
    func, *arg = args
    return func(*arg)


# 封装Chem.MolFromSmiles(smiles)
def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)


# 封装Chem.MolToSmiles(smiles)
def mol_to_smiles(smiles):
    return Chem.MolToSmiles(smiles)


# 封装Chem.AddHs(mol)
def add_Hs(mol):
    return Chem.AddHs(mol)


# 数据预处理
def preprocess(file_path, output_file_path, mw_low, mw_high, acid_or_base, is_input_file=True, is_output_file=True):
    if is_input_file:
        df = pd.read_csv(file_path)
    else:
        df = file_path

    # 创建一个进程池
    with Pool(processes=cpu_count()) as pool:
        # 标准化smiles
        df['Smiles'] = pool.map(worker, [(standardize_smiles, x) for x in df['Smiles']])
        print('标准化SMILES：', df.shape)

        # 删除重复分子
        df = df.drop_duplicates(subset='Smiles', keep=False)
        print('删除重复分子：', df.shape)

        # 创建mol列
        df['mol'] = pool.map(worker, [(mol_from_smiles, x) for x in df['Smiles']])
        print('创建mol列：', df.shape)

        # 补氢
        df['mol'] = pool.map(worker, [(add_Hs, x) for x in df['mol']])
        df['Smiles'] = pool.map(worker, [(mol_to_smiles, x) for x in df['mol']])
        print('补氢：', df.shape)

        # 删除无机物分子
        df = df[pd.Series(pool.map(worker, [(is_inorganic_substance, x) for x in df['mol']]), index=df.index) == False]
        print('删除无机物分子：', df.shape)

        # 删除混合物
        df = df[pd.Series(pool.map(worker, [(is_mixture, x) for x in df['mol']]), index=df.index) == False]
        print('删除混合物：', df.shape)

        # 删除不符合分子量要求的分子
        df = df[pd.Series(pool.map(worker, [(is_molecular_weight_moderate, x, mw_low, mw_high) for x in df['mol']]),
                          index=df.index)]
        print('删除不符合分子量要求的分子：', df.shape)

        # 删除不含可识别电离位点的分子
        df = df[pd.Series(pool.map(worker,
                                   [(is_recognizable_ionization_sites_contained, x, smarts_file_path, acid_or_base) for
                                    x in df['mol']]), index=df.index)]
        print('删除不含可识别电离位点的分子：', df.shape)

        # 后续处理，用于保存文件
        df = df.drop('mol', axis=1)  # 删除mol列，只保留其他三列
        df.index = [x for x in range(df.shape[0])]
        if is_output_file:
            df.to_csv(output_file_path)  # 将处理后的结果保存到output_file_path路径下
        else:
            return df


# 酸碱分离
def split_acid_base(file_path, acid_save_path, base_save_path):
    data = pd.read_csv(file_path)
    acid = data[data['basicOrAcidic'] == 'acidic']
    base = data[data['basicOrAcidic'] == 'basic']
    acid.to_csv(acid_save_path)
    base.to_csv(base_save_path)


# 获取当前时间
def get_time_now():
    # 获取当前时间
    current_time = datetime.now()
    # 将时间格式化为字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"当前时间是: {formatted_time}")



# 二、构建图后的数据处理
# 读取pt文件
def load_pt(pt_file_path):
    pt = torch.load(pt_file_path)
    return pt


# 以DataCollate对象的形式保存pt文件，主要包含两个属性，分别为Data对象及其切片字典
def save_pt(data_col, pt_save_path):
    torch.save(data_col, pt_save_path)


# 定义DataCollate类，用于管理data和slices数据
class DataCollate(InMemoryDataset, ABC):
    def __init__(self, data):
        super().__init__()
        self.data, self.slices = data


# 创建DataCollate对象
def get_DataCollate(_data, _slice):
    data_col = DataCollate((_data, _slice))
    return data_col


# 划分训练集和内部验证集,data_col是由多个Data对象构成的，其中每个元素对应一个分子
def split_train_valid(data_col, valid_size=1 / 9, save_pt_file=True, if_split_acid_base=False, acid_or_base=''):
    data_train, data_valid = train_test_split(data_col, test_size=valid_size, shuffle=True, random_state=42)
    # print(data_col)
    # print(len(data_train))
    # print(len(data_valid))
    imd = InMemoryDataset()
    data_train, slice_train = imd.collate(data_train)
    data_valid, slice_valid = imd.collate(data_valid)
    data_train = get_DataCollate(data_train, slice_train)
    data_valid = get_DataCollate(data_valid, slice_valid)
    if if_split_acid_base:
        if save_pt_file:
            save_pt(data_train,
                    'division_results/division_results_' + str(result_num) + '/data_' + acid_or_base + '_train.pt')
            save_pt(data_valid,
                    'division_results/division_results_' + str(result_num) + '/data_' + acid_or_base + '_valid.pt')
    else:
        if save_pt_file:
            save_pt(data_train, 'division_results/division_results_' + str(result_num) + '/data_train.pt')
            save_pt(data_valid, 'division_results/division_results_' + str(result_num) + '/data_valid.pt')
    return data_train, data_valid


# 划分训练集（包括训练集和内部验证集，还需要进一步划分）和测试集,data_col是由多个Data对象构成的，其中每个元素对应一个分子
def split_train_test(data_col, test_size=0.1, save_pt_file=True, if_split_acid_base=False, acid_or_base=''):
    data_train, data_test = train_test_split(data_col, test_size=test_size, shuffle=True, random_state=42)
    # print(data_col)
    # print(len(data_train))
    # print(len(data_test))
    imd = InMemoryDataset()
    data_train, slice_train = imd.collate(data_train)
    data_test, slice_test = imd.collate(data_test)
    data_train = get_DataCollate(data_train, slice_train)
    data_test = get_DataCollate(data_test, slice_test)
    if if_split_acid_base:
        if save_pt_file:
            save_pt(data_train,
                    'division_results/division_results_' + str(
                        result_num) + '/data_' + acid_or_base + '_train_valid.pt')
            save_pt(data_test,
                    'division_results/division_results_' + str(result_num) + '/data_' + acid_or_base + '_test.pt')
    else:
        if save_pt_file:
            save_pt(data_train, 'division_results/division_results_' + str(result_num) + '/data_train_valid.pt')
            save_pt(data_test, 'division_results/division_results_' + str(result_num) + '/data_test.pt')
    return data_train, data_test


def split_train_valid_test(data, acid_or_base, valid_size=0.1, test_size=0.1, cv=5):
    for i in range(cv):
        data_train_valid, data_test = train_test_split(data, test_size=test_size, shuffle=True)
        data_train, data_valid = train_test_split(data_train_valid, test_size=valid_size / (1 - test_size),
                                                  shuffle=True)
        if acid_or_base == 'acid':
            data_train.to_csv(
                acid_train_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + acid_train_save_path.split('.')[1])
            data_valid.to_csv(
                acid_valid_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + acid_valid_save_path.split('.')[1])
            data_test.to_csv(
                acid_test_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + acid_test_save_path.split('.')[1])
        elif acid_or_base == 'base':
            data_train.to_csv(
                base_train_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + base_train_save_path.split('.')[1])
            data_valid.to_csv(
                base_valid_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + base_valid_save_path.split('.')[1])
            data_test.to_csv(
                base_test_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + base_test_save_path.split('.')[1])
        else:
            print('The parameter acid_or_base setting is incorrect. It can only be \'acid\' or \'base\'.')


def batch_get_graphs(cv):
    for i in range(cv):
        print('Random split ', i + 1, ' / ', cv, ':', sep='')
        _data_acid_train, _slice_acid_train = get_graph(
            acid_train_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + acid_train_save_path.split('.')[1])
        _data_acid_valid, _slice_acid_valid = get_graph(
            acid_valid_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + acid_valid_save_path.split('.')[1])
        _data_acid_test, _slice_acid_test = get_graph(
            acid_test_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + acid_test_save_path.split('.')[1])
        _data_base_train, _slice_base_train = get_graph(
            base_train_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + base_train_save_path.split('.')[1])
        _data_base_valid, _slice_base_valid = get_graph(
            base_valid_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + base_valid_save_path.split('.')[1])
        _data_base_test, _slice_base_test = get_graph(
            base_test_save_path.split('.')[0] + '_random_' + str(i + 1) + '.' + base_test_save_path.split('.')[1])
        data_col_acid_train = get_DataCollate(_data_acid_train, _slice_acid_train)
        data_col_acid_valid = get_DataCollate(_data_acid_valid, _slice_acid_valid)
        data_col_acid_test = get_DataCollate(_data_acid_test, _slice_acid_test)
        data_col_base_train = get_DataCollate(_data_base_train, _slice_base_train)
        data_col_base_valid = get_DataCollate(_data_base_valid, _slice_base_valid)
        data_col_base_test = get_DataCollate(_data_base_test, _slice_base_test)
        save_pt(data_col_acid_train,
                'division_results/division_results_' + str(result_num) + '/data_acid_train_random_' + str(
                    i + 1) + '.pt')
        save_pt(data_col_acid_valid,
                'division_results/division_results_' + str(result_num) + '/data_acid_valid_random_' + str(
                    i + 1) + '.pt')
        save_pt(data_col_acid_test,
                'division_results/division_results_' + str(result_num) + '/data_acid_test_random_' + str(
                    i + 1) + '.pt')
        save_pt(data_col_base_train,
                'division_results/division_results_' + str(result_num) + '/data_base_train_random_' + str(
                    i + 1) + '.pt')
        save_pt(data_col_base_valid,
                'division_results/division_results_' + str(result_num) + '/data_base_valid_random_' + str(
                    i + 1) + '.pt')
        save_pt(data_col_base_test,
                'division_results/division_results_' + str(result_num) + '/data_base_test_random_' + str(
                    i + 1) + '.pt')


def split_train_valid_test_in_order(data, acid_or_base=''):
    data = data.sort_values(by='pKa', ascending=True)[['Smiles', 'basicOrAcidic', 'pKa']].reset_index(
        drop=True)  # 按照pKa升序排序
    set_name = []
    # 划分数据集
    for i in range(data.shape[0]):
        if (i - 3) % 10 == 0 or (i - 4) % 10 == 0 or (i - 5) % 10 == 0 or (i - 6) % 10 == 0 or (
                i - 7) % 10 == 0 or (
                i - 8) % 10 == 0 or (i - 9) % 10 == 0 or (i - 10) % 10 == 0:
            set_name.append('train')  # 训练集数据
        elif (i - 2) % 10 == 0:
            set_name.append('valid')  # 验证集数据
        elif (i - 1) % 10 == 0:
            set_name.append('test')  # 测试集数据
        else:
            print('Error!')
    data['set_name'] = set_name
    data_train = data[data['set_name'] == 'train'].sample(frac=1, random_state=42)
    data_valid = data[data['set_name'] == 'valid'].sample(frac=1, random_state=42)
    data_test = data[data['set_name'] == 'test'].sample(frac=1, random_state=42)
    # 保存文件
    if acid_or_base == 'acid':
        data_train.to_csv(acid_train_save_path)
        data_valid.to_csv(acid_valid_save_path)
        data_test.to_csv(acid_test_save_path)
    elif acid_or_base == 'base':
        data_train.to_csv(base_train_save_path)
        data_valid.to_csv(base_valid_save_path)
        data_test.to_csv(base_test_save_path)
    else:
        print('Error!')
        return 0


# 获取attentiveFP中forword函数的batch中的一批，_slice是DataCollate中的切片信息，x_size是每一个批次中的原子数（即原子特征矩阵的行数），
# begin_index是本批次的首个原子的index，molecule_index是本批次的首个分子的index
def get_batch(_slices, x_size, begin_index, molecule_index):
    batch = []
    num = 0
    for i in range(begin_index, begin_index + x_size):
        if i == _slices[molecule_index + 1]:
            molecule_index += 1
            num += 1
        batch.append(num)
    return torch.tensor(batch), begin_index + x_size, molecule_index + 1


# 导入数据，pt_file_path_list为多个pt文件路径构成的列表，用于导入多次划分训练、验证和测试集的单次结果
def load_data(pt_file_path_list):
    data = []
    _slice_list = []
    copy_list = []
    for pt_file_path in pt_file_path_list:
        data_i = load_pt(pt_file_path=pt_file_path)
        data.append(data_i)
        _slice_list.append(data_i.slices['x'])
        copy_list.append(copy.deepcopy(data_i))
    return *data, *_slice_list, *copy_list


# 批量导入数据，用于导入多次划分训练、验证和测试集的所有结果
def batch_load_data(pt_file_path_list):
    data_train, data_valid, data_test, _slice_train, _slice_valid, _slice_test, data_train_copy, data_valid_copy, data_test_copy = [], [], [], [], [], [], [], [], []
    for pt_file_path_list_i in pt_file_path_list:
        data_train_i, data_valid_i, data_test_i, _slice_train_i, _slice_valid_i, _slice_test_i, data_train_copy_i, data_valid_copy_i, data_test_copy_i = load_data(
            pt_file_path_list_i)
        data_train.append(data_train_i)
        data_valid.append(data_valid_i)
        data_test.append(data_test_i)
        _slice_train.append(_slice_train_i)
        _slice_valid.append(_slice_valid_i)
        _slice_test.append(_slice_test_i)
        data_train_copy.append(data_train_copy_i)
        data_valid_copy.append(data_valid_copy_i)
        data_test_copy.append(data_test_copy_i)
    return data_train, data_valid, data_test, _slice_train, _slice_valid, _slice_test, data_train_copy, data_valid_copy, data_test_copy


# 数据分批,返回的三个值都是list，每个list包括cv个数据，每个数据是每次划分的训练、验证或测试集
def batch_data(data_train, data_valid, data_test):
    data_train_list, data_valid_list, data_test_list = [], [], []
    for data_train_i, data_valid_i, data_test_i in zip(data_train, data_valid, data_test):
        data_train_i = DataLoader(dataset=data_train_i, batch_size=batch_size, shuffle=False, drop_last=False,
                                  pin_memory=True)
        data_valid_i = DataLoader(dataset=data_valid_i, batch_size=batch_size, shuffle=False, drop_last=False,
                                  pin_memory=True)
        data_test_i = DataLoader(dataset=data_test_i, batch_size=batch_size, shuffle=False, drop_last=False,
                                 pin_memory=True)
        data_train_list.append(data_train_i)
        data_valid_list.append(data_valid_i)
        data_test_list.append(data_test_i)

    return data_train_list, data_valid_list, data_test_list


# 将结果输出为csv
# indicators_values_list是n行m列的list或np.ndarray，n为进行的epoches数，m为评价指标数，
# indicators_names_list是长度为m的列表，是上述指标的名称，需要与上面的指标一一对应
# indicators_table_save_path为结果的保存路径
def save_results_as_csv(indicators_values_list, indicators_names_list, indicators_table_save_path):
    indicators_table = pd.DataFrame(indicators_values_list, index=range(1, indicators_values_list.shape[0] + 1),
                                    columns=indicators_names_list)
    indicators_table.index.name = 'epoch'
    indicators_table.to_csv(indicators_table_save_path)


# 获取FPGNN模型提取的嵌入
def get_model_embeddings(model, data, devices):
    # 定义一个钩子函数
    def hook(module, input, output):
        global embeddings
        embeddings = output.detach()

    # 定义一个全局变量来保存嵌入
    global embeddings
    embeddings = None

    # 在你想要获取输出的层上注册钩子
    handle = model.FFN[3].register_forward_hook(hook)

    embeddings_all = torch.Tensor().to(devices)
    # 运行你的模型
    for index, data_i in enumerate(data):
        if index % 1 == 0:
            print(index)
        # 将数据移动到设备上
        data_i = data_i.to(devices)
        # 前向传播，得到预测值
        output = model(smiles_list=data_i.Smiles, x=data_i.x, edge_index=data_i.edge_index,
                       edge_attr=data_i.edge_feature, batch=data_i.batch)
        # 将当前循环的embeddings添加到embeddings_all中
        embeddings_all = torch.cat((embeddings_all, embeddings), dim=0)

    # 移除钩子
    handle.remove()

    return embeddings_all


if __name__ == '__main__':
    # 获取当前时间
    get_time_now()

    # 根据数据集的basicOrAcidic列划分酸碱
    print('Acid and base splitting....')
    split_acid_base(csv_file_path, acid_save_path, base_save_path)
    # 预处理
    print('Acid preprocess....')
    preprocess(acid_save_path, acid_save_path, mw_low, mw_high, 'acid')  # 对酸性物质进行预处理
    print('Base preprocess....')
    preprocess(base_save_path, base_save_path, mw_low, mw_high, 'base')  # 对碱性物质进行预处理
    # 划分数据集
    print('Training set and test set splitting....')
    acid_data = pd.read_csv(acid_save_path)
    base_data = pd.read_csv(base_save_path)
    split_train_valid_test(data=acid_data, acid_or_base='acid', valid_size=0.1, test_size=0.1, cv=cv)
    split_train_valid_test(data=base_data, acid_or_base='base', valid_size=0.1, test_size=0.1, cv=cv)
    # 构建图结构并保存
    print('Graph creating....')
    batch_get_graphs(cv=cv)

    # 获取当前时间
    get_time_now()