# 导入需要的库
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from training_block import *
from model_params import *
from data_processing import DataCollate, get_time_now, batch_load_data, batch_data, load_pt
from torch.nn.functional import mse_loss
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def main():
    get_time_now()  # 获取当前时间
    _slice_ext, data_ext = [], []
    if acid_or_base == 'acid':
        ext_path_list = ['input/ext_set/sampl6_acidic.pt', 'input/ext_set/sampl7_acidic.pt',
                         'input/ext_set/organic_acidic.pt', 'input/ext_set/drug_acidic.pt']
    elif acid_or_base == 'base':
        ext_path_list = ['input/ext_set/sampl6_basic.pt', 'input/ext_set/organic_basic.pt',
                         'input/ext_set/drug_basic.pt']
    for path in ext_path_list:
        _slice_ext.append(load_pt(path).slices['x'])
        data_ext.append(
            DataLoader(dataset=load_pt(path), batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True))

    # 导入数据
    print('Data loading....')
    pt_flie_path_list = [['division_results/division_results_' + str(result_num) + '/data_' +
                          acid_or_base + '_train_random_' + str(i + 1) + '.pt',
                          'division_results/division_results_' + str(result_num) + '/data_' +
                          acid_or_base + '_valid_random_' + str(i + 1) + '.pt',
                          'division_results/division_results_' + str(result_num) + '/data_' +
                          acid_or_base + '_test_random_' + str(i + 1) + '.pt']
                         for i in range(cv)]
    data_train, data_valid, data_test, _slice_train, _slice_valid, _slice_test, data_train_copy, data_valid_copy, data_test_copy = batch_load_data(
        pt_flie_path_list)

    # 数据分批
    print('Data batching....')
    data_train, data_valid, data_test = batch_data(data_train, data_valid, data_test)

    # 导入预训练后的模型
    print('Model loading....')
    model = torch.load('model/model_35/FPGNN_base_macro_finetuning_model_e_025.pth')
    model.to(devices)  # 将模型移动到设备上，如果有GPU则使用GPU，否则使用CPU
    model = [model] * cv

    # 创建优化器对象，使用Adam优化器，输入参数为模型参数，学习率和权重衰减系数，定义损失函数
    print('Optimizer creating....')
    optimizer = [optim.Adam(model_i.parameters(), lr=lr, weight_decay=weight_decay) for model_i in model]  # 优化器
    loss_function = mse_loss  # 损失函数
    get_time_now()  # 获取当前时间

    # 训练与评价模型
    print('Begin training.')
    train_and_eval(
        model=model,
        data_train=data_train,
        data_valid=data_valid,
        data_test=data_test,
        data_ext=data_ext,
        _slice_train=_slice_train,
        _slice_valid=_slice_valid,
        _slice_test=_slice_test,
        _slice_ext=_slice_ext,
        optimizer=optimizer,
        loss_function=loss_function,
        epochs=epochs,
        devices=devices,
        eval_interval=1)
    get_time_now()  # 获取当前时间
    print('Finished.')


if __name__ == '__main__':
    main()
