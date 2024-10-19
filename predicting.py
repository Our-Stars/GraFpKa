import numpy as np
import pandas as pd
import argparse
from data_processing import load_pt, get_DataCollate, save_pt, is_recognizable_ionization_sites_contained
from model_params import devices, batch_size
from graph_creating import get_graph
from torch import load
from torch_geometric.data import DataLoader
import warnings
from rdkit import Chem
from model_params import smarts_file_path

warnings.filterwarnings("ignore", message="'data.DataLoader' is deprecated, use 'loader.DataLoader' instead",
                        category=UserWarning)


# 获取具有可电离位点的SMILES的01字符串
def get_ionizable_smiles_binary(smiles_list, smarts_file_path, acid_or_base):
    ionizable_binary = [0] * len(smiles_list)  # 初始化一个与smiles_list长度相同的全零列表
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES at index {i}: {smiles}")
            continue
        mol = Chem.AddHs(mol)  # 为分子添加氢原子
        # 判断分子是否含有可电离位点
        if is_recognizable_ionization_sites_contained(mol, smarts_file_path, acid_or_base):
            ionizable_binary[i] = 1  # 如果含有可电离位点，则将对应位置的值设为1
    return ionizable_binary


def predict_pKa(smiles_list, ionizable_binary, model):
    # 构造pKa和酸碱性列表，初始化为None占位符
    pka_list = [None] * len(smiles_list)

    # 仅处理含有可电离位点的SMILES
    smiles_list_to_predict = [smiles_list[i] for i in range(len(smiles_list)) if ionizable_binary[i] == 1]

    # 如果没有需要预测的SMILES，直接返回占位符
    if len(smiles_list_to_predict) == 0:
        return pka_list

    # 定义Series
    smiles_series = pd.Series(smiles_list_to_predict, name='Smiles')
    pka_series = pd.Series([0.0] * len(smiles_list_to_predict), name='pKa')
    basicOrAcidic_series = pd.Series(['XXX'] * len(smiles_list_to_predict), name='basicOrAcidic')

    # 定义DataFrame并写入文件
    data = pd.concat([smiles_series, pka_series, basicOrAcidic_series], axis=1)
    data.to_csv('temp/data_predicting.csv', index=False)

    # 构造分子图并保存
    data, slice = get_graph('temp/data_predicting.csv')
    data = get_DataCollate(data, slice)
    save_pt(data, 'temp/data_predicting.pt')

    # 数据分批并输出预测值
    data = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    predictions = []
    for index, data_i in enumerate(data):
        # 将数据移动到设备上
        data_i = data_i.to(devices)
        output = model(smiles_list=data_i.Smiles, x=data_i.x, edge_index=data_i.edge_index,
                       edge_attr=data_i.edge_feature, batch=data_i.batch).tolist()
        predictions.extend(np.array(output).flatten().tolist())

    # 将预测结果填充回pka_list
    pred_index = 0
    for i in range(len(smiles_list)):
        if ionizable_binary[i] == 1:
            pka_list[i] = predictions[pred_index]
            pred_index += 1

    return pka_list


if __name__ == '__main__':
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(description="pKa Prediction Script")

    # 互斥组，确保只能选择-s或-i
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--s', nargs='+', help="List of SMILES strings to predict")
    group.add_argument('--i', type=str, help="Path to a CSV file containing SMILES strings (one per line, no header)")
    parser.add_argument('--o', default='results/predicted_results.csv',
                        help="Path to save the output CSV file (default: results/predicted_results.csv)")
    args = parser.parse_args()

    # 根据输入方式获取SMILES列表
    if args.s:
        smiles_list = args.s
    elif args.i:
        # 从CSV文件中读取SMILES列表
        smiles_df = pd.read_csv(args.i, header=None)  # 读取没有表头的CSV
        smiles_list = smiles_df[0].tolist()  # 获取第一列的数据作为SMILES列表

    # 获取具有可电离位点的SMILES的01字符串
    ionizable_binary_acidic = get_ionizable_smiles_binary(smiles_list, smarts_file_path, 'acid')
    ionizable_binary_basic = get_ionizable_smiles_binary(smiles_list, smarts_file_path, 'base')

    # 导入模型
    model_acidic = load(r'model/GraFpKa_acidic_model.pth').to(devices)
    model_basic = load(r'model/GraFpKa_basic_model.pth').to(devices)

    # 预测pKa，保持与原始SMILES列表的对应关系
    pKa_acidic = predict_pKa(smiles_list, ionizable_binary_acidic, model_acidic)
    pKa_basic = predict_pKa(smiles_list, ionizable_binary_basic, model_basic)

    # 保存预测结果
    print('Predicted results:')
    predicted_dict = {'SMILES': smiles_list, 'predicted acidic pKa': pKa_acidic, 'predicted basic pKa': pKa_basic}
    predicted_df = pd.DataFrame(predicted_dict)
    predicted_df.to_csv(args.o, index=False)
    print(predicted_df)
