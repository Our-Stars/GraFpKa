import os
import numpy as np
import pandas as pd
import argparse
from plot_drawing import batch_highlight_atoms_or_bonds_from_IG
from data_processing import load_pt, get_DataCollate, save_pt, is_recognizable_ionization_sites_contained
from model_params import devices, batch_size, smarts_file_path
from graph_creating import get_graph
from torch import load
from rdkit import Chem
import multiprocessing


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


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(description="SMILES Interpretability Script")

    # 互斥组，确保只能选择-s或-i
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--s', nargs='+', help="List of SMILES strings to predict")
    group.add_argument('--i', type=str, help="Path to a CSV file containing SMILES strings (one per line, no header)")

    # 可选参数：指定输出文件夹，确保以'/'结尾
    parser.add_argument('--o', default='visualization/', help="Path to save the output images, must end with '/' (default: visualization/)")

    # threshold 可选参数，默认为 -1
    parser.add_argument('--t', type=float, default=-1, help="Threshold for color bar limits during visualization (default: -1 for automatic)")

    args = parser.parse_args()

    # 验证输出路径是否以 '/' 结尾
    if not args.o.endswith('/'):
        raise ValueError("The output folder path must end with '/'.")

    # 根据输入方式获取SMILES列表
    if args.s:
        smiles_list = args.s
    elif args.i:
        # 从CSV文件中读取SMILES列表
        smiles_df = pd.read_csv(args.i, header=None)  # 读取没有表头的CSV
        smiles_list = smiles_df[0].tolist()  # 获取第一列的数据作为SMILES列表

    # 定义模型
    model_acidic = load(r'model/GraFpKa_acidic_model.pth').to(devices)
    model_basic = load(r'model/GraFpKa_basic_model.pth').to(devices)

    # 获取具有可电离位点的SMILES的01字符串
    ionizable_binary_acidic = get_ionizable_smiles_binary(smiles_list, smarts_file_path, 'acid')
    ionizable_binary_basic = get_ionizable_smiles_binary(smiles_list, smarts_file_path, 'base')

    # 根据01字符串筛选出含有酸性或碱性可电离位点的SMILES
    smiles_list_acidic = [smiles_list[i] for i in range(len(smiles_list)) if ionizable_binary_acidic[i] == 1]
    smiles_list_basic = [smiles_list[i] for i in range(len(smiles_list)) if ionizable_binary_basic[i] == 1]

    # 构造pKa和酸碱性列表
    pka_list_acidic = [0.0] * len(smiles_list_acidic)
    pka_list_basic = [0.0] * len(smiles_list_basic)

    # 处理并可视化 - Acidic
    if smiles_list_acidic:
        smiles_series_acidic = pd.Series(smiles_list_acidic, name='Smiles')
        pka_series_acidic = pd.Series(pka_list_acidic, name='pKa')
        basicOrAcidic_series_acidic = pd.Series(['acidic'] * len(smiles_list_acidic), name='basicOrAcidic')

        data_acidic = pd.concat([smiles_series_acidic, pka_series_acidic, basicOrAcidic_series_acidic], axis=1)
        data_acidic.to_csv('temp/data_acidic_interpretability.csv', index=False)

        # 构造分子图并保存
        data_acidic, slice_acidic = get_graph('temp/data_acidic_interpretability.csv')
        data_acidic = get_DataCollate(data_acidic, slice_acidic)
        save_pt(data_acidic, 'temp/data_acidic_interpretability.pt')

        # 使用IG进行酸性可电离位点的可视化
        batch_highlight_atoms_or_bonds_from_IG(data_acidic, model_acidic, args.o, devices, 'acidic', args.t)

    # 处理并可视化 - Basic
    if smiles_list_basic:
        smiles_series_basic = pd.Series(smiles_list_basic, name='Smiles')
        pka_series_basic = pd.Series(pka_list_basic, name='pKa')
        basicOrAcidic_series_basic = pd.Series(['basic'] * len(smiles_list_basic), name='basicOrAcidic')

        data_basic = pd.concat([smiles_series_basic, pka_series_basic, basicOrAcidic_series_basic], axis=1)
        data_basic.to_csv('temp/data_basic_interpretability.csv', index=False)

        # 构造分子图并保存
        data_basic, slice_basic = get_graph('temp/data_basic_interpretability.csv')
        data_basic = get_DataCollate(data_basic, slice_basic)
        save_pt(data_basic, 'temp/data_basic_interpretability.pt')

        # 使用IG进行碱性可电离位点的可视化
        batch_highlight_atoms_or_bonds_from_IG(data_basic, model_basic, args.o, devices, 'basic', args.t)
