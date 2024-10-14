import numpy as np
from data_processing import load_pt, get_DataCollate, DataCollate, save_pt
from model_params import devices, batch_size
from graph_creating import get_graph
from torch import load
import pandas as pd
from torch_geometric.data import DataLoader
import warnings

warnings.filterwarnings("ignore", message="'data.DataLoader' is deprecated, use 'loader.DataLoader' instead",
                        category=UserWarning)
if __name__ == '__main__':
    # 井号之间为可以修改的参数部分
    #################################################################
    # 在此处输入需要预测的分子的SMILES式列表并选择需要预测酸性pKa还是碱性pKa，预测pKa顺序与此处的SMILES顺序一致
    smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'Cc1ccccc1NN=C(C#N)C#N', 'COc1cccc(S(N)(=O)=O)c1']
    # 可以填写acidic或者basic，表示需要预测酸性pKa还是碱性pKa
    acidic_or_basic = 'acidic'
    #################################################################

    # 输入检查
    assert acidic_or_basic == 'acidic' or acidic_or_basic == 'basic', '\'acidic_or_basic\' can only be filled with either \'acidic\' or \'basic\'.'

    # 导入模型
    model_acidic = load(r'model/GraFpKa_acidic_model.pth').to(devices)
    model_basic = load(r'model/GraFpKa_basic_model.pth').to(devices)

    # 构造pKa和酸碱性列表
    pka_list = [0.0] * len(smiles_list)
    basicOrAcidic_list = ['XXX'] * len(smiles_list)

    # 定义Series
    smiles_series = pd.Series(smiles_list, name='Smiles')
    pka_series = pd.Series(pka_list, name='pKa')
    basicOrAcidic_series = pd.Series(basicOrAcidic_list, name='basicOrAcidic')

    # 定义DataFrame并写入文件
    data = pd.concat([smiles_series, pka_series, basicOrAcidic_series], axis=1)
    data.to_csv('temp/data_predicting.csv', index=False)

    # 构造分子图并保存
    data, slice = get_graph('temp/data_predicting.csv')
    data = get_DataCollate(data, slice)
    save_pt(data, 'temp/data_predicting.pt')

    # 数据分批并输出预测值
    data = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    for index, data_i in enumerate(data):
        # 将数据移动到设备上
        data_i = data_i.to(devices)
        # 前向传播，得到预测值
        if acidic_or_basic == 'acidic':
            output = model_acidic(smiles_list=data_i.Smiles, x=data_i.x, edge_index=data_i.edge_index,
                                  edge_attr=data_i.edge_feature, batch=data_i.batch).tolist()
        elif acidic_or_basic == 'basic':
            output = model_basic(smiles_list=data_i.Smiles, x=data_i.x, edge_index=data_i.edge_index,
                                 edge_attr=data_i.edge_feature, batch=data_i.batch).tolist()

    # 输出预测结果，顺序与smiles_list中的一致
    output = np.array(output).flatten().tolist()
    print(output)
