from plot_drawing import batch_highlight_atoms_or_bonds_from_IG
from data_processing import load_pt, get_DataCollate, DataCollate, save_pt
from model_params import devices, batch_size
from graph_creating import get_graph
from torch import load
import pandas as pd
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 井号之间为可以修改的参数部分，可视化结果在“visualization”文件夹下
    #################################################################
    # 需要进行可解释性分析的SMILES列表，根据需要进行修改即可
    smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'Cc1ccccc1NN=C(C#N)C#N', 'COc1cccc(S(N)(=O)=O)c1']
    # 可以填写acidic或者basic，表示需要预测酸性pKa还是碱性pKa
    acidic_or_basic = 'acidic'
    # threshold为绘图时右侧彩条坐标的上限值（下限值的相反数），-1表示自动设置上下限
    # 需要对比多个化合物的可视化结果时，建议手动设置上下限，可以统一多个图的上下限，便于比较
    threshold = -1
    #################################################################

    # 输入检查
    assert acidic_or_basic == 'acidic' or acidic_or_basic == 'basic', '\'acidic_or_basic\' can only be filled with either \'acidic\' or \'basic\'.'

    # 构造pKa和酸碱性列表
    pka_list = [0.0] * len(smiles_list)
    basicOrAcidic_list = ['XXX'] * len(smiles_list)

    # 定义模型，根据需要进行修改即可
    model_acidic = load(r'model/GraFpKa_acidic_model.pth').to(devices)
    model_basic = load(r'model/GraFpKa_basic_model.pth').to(devices)

    # 定义Series
    smiles_series = pd.Series(smiles_list, name='Smiles')
    pka_series = pd.Series(pka_list, name='pKa')
    basicOrAcidic_series = pd.Series(basicOrAcidic_list, name='basicOrAcidic')

    # 定义DataFrame并写入文件
    data = pd.concat([smiles_series, pka_series, basicOrAcidic_series], axis=1)
    data.to_csv('temp/data_interpretability.csv', index=False)

    # 构造分子图并保存
    data, slice = get_graph('temp/data_interpretability.csv')
    data = get_DataCollate(data, slice)
    save_pt(data, 'temp/data_interpretability.pt')

    # 使用IG进行可视化
    if acidic_or_basic == 'acidic':
        batch_highlight_atoms_or_bonds_from_IG(data, model_acidic, 'visualization/', devices, threshold)
    elif acidic_or_basic == 'basic':
        batch_highlight_atoms_or_bonds_from_IG(data, model_basic, 'visualization/', devices, threshold)
