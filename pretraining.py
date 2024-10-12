# 导入需要的库
import pandas as pd
import torch.optim as optim
from graph_creating import get_graph, is_ionization_center_all
from torch_geometric.data import Data, DataLoader
from training_block import *
from model_params import *
from data_processing import load_data, save_results_as_csv, DataCollate, get_time_now, load_pt
from torch.nn.functional import mse_loss
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def main():
    get_time_now()  # 获取当前时间
    data_ext = load_pt('input/ext_set/sampl_acidic.pt')
    _slice_ext = data_ext.slices['x']
    data_ext = DataLoader(dataset=data_ext, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    # 导入数据
    print('Data loading....')
    pt_flie_path_list = [
        'division_results/division_results_' + str(result_num) + '/data_' + acid_or_base + '_train_random_1.pt',
        'division_results/division_results_' + str(result_num) + '/data_' + acid_or_base + '_valid_random_1.pt',
        'division_results/division_results_' + str(result_num) + '/data_' + acid_or_base + '_test_random_1.pt']
    data_train, data_valid, data_test, _slice_train, _slice_valid, _slice_test, data_train_copy, data_valid_copy, data_test_copy = load_data(
        pt_flie_path_list)

    # 数据分批
    print('Data batching....')
    data_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    data_valid = DataLoader(dataset=data_valid, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    data_test = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # 计算或导入每个批次的电离中心布尔列表
    if macro_or_micro == 'micro':
        if cal_ionization_center_bool_list:
            smarts_df = pd.read_csv(smarts_file_path, delimiter='\t')
            ionization_center_bool_list_train = is_ionization_center_all(data=data_train,
                                                                         smarts_df=smarts_df,
                                                                         acid_or_base=acid_or_base,
                                                                         save_path='input/ionization_center_bool_list/ionization_center_bool_list_' + acid_or_base + '_train.pt',
                                                                         verbose=True)
            ionization_center_bool_list_valid = is_ionization_center_all(data=data_valid,
                                                                         smarts_df=smarts_df,
                                                                         acid_or_base=acid_or_base,
                                                                         save_path='input/ionization_center_bool_list/ionization_center_bool_list_' + acid_or_base + '_valid.pt',
                                                                         verbose=True)
            ionization_center_bool_list_test = is_ionization_center_all(data=data_test,
                                                                        smarts_df=smarts_df,
                                                                        acid_or_base=acid_or_base,
                                                                        save_path='input/ionization_center_bool_list/ionization_center_bool_list_' + acid_or_base + '_test.pt',
                                                                        verbose=True)
            ionization_center_bool_list_ext = is_ionization_center_all(data=data_ext,
                                                                       smarts_df=smarts_df,
                                                                       acid_or_base=acid_or_base,
                                                                       save_path='input/ionization_center_bool_list/ionization_center_bool_list_' + acid_or_base + '_ext.pt',
                                                                       verbose=True)
        else:
            ionization_center_bool_list_train = torch.load(
                'input/ionization_center_bool_list/ionization_center_bool_list_' + acid_or_base + '_train.pt')
            ionization_center_bool_list_valid = torch.load(
                'input/ionization_center_bool_list/ionization_center_bool_list_' + acid_or_base + '_valid.pt')
            ionization_center_bool_list_test = torch.load(
                'input/ionization_center_bool_list/ionization_center_bool_list_' + acid_or_base + '_test.pt')
            ionization_center_bool_list_ext = torch.load(
                'input/ionization_center_bool_list/ionization_center_bool_list_' + acid_or_base + '_ext.pt')
    elif macro_or_micro == 'macro':
        ionization_center_bool_list_train = None
        ionization_center_bool_list_valid = None
        ionization_center_bool_list_test = None
        ionization_center_bool_list_ext = None

    # 创建模型对象，输入参数为原子特征维度，隐藏层维度和输出层维度
    print('Model initializing....')
    model_class = model_dict[model_name]['class']
    model_params = model_dict[model_name]['params']
    model = model_class(**model_params)
    model.to(devices)  # 将模型移动到设备上，如果有GPU则使用GPU，否则使用CPU

    # 创建优化器对象，使用Adam优化器，输入参数为模型参数，学习率和权重衰减系数，定义损失函数
    print('Optimizer creating....')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 优化器
    loss_function = mse_loss  # 损失函数
    get_time_now()  # 获取当前时间

    # 训练与评价模型
    print('Begin training.')
    R2_train_list, R2_valid_list, R2_test_list, MAE_train_list, MAE_valid_list, MAE_test_list, RMSE_train_list, RMSE_valid_list, RMSE_test_list = train_and_eval(
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
        macro_or_micro=macro_or_micro,
        ionization_center_bool_list_train=ionization_center_bool_list_train,
        ionization_center_bool_list_valid=ionization_center_bool_list_valid,
        ionization_center_bool_list_test=ionization_center_bool_list_test,
        ionization_center_bool_list_ext=ionization_center_bool_list_ext,
        eval_interval=1)
    get_time_now()  # 获取当前时间

    # 输出每一轮指标的csv文件
    print('Indicators summarizing....')
    indicators = np.array([R2_train_list, R2_valid_list, R2_test_list, MAE_train_list, MAE_valid_list, MAE_test_list,
                           RMSE_train_list, RMSE_valid_list, RMSE_test_list]).T
    columns_names = ['R2_train', 'R2_valid', 'R2_test', 'MAE_train', 'MAE_valid', 'MAE_test', 'RMSE_train',
                     'RMSE_valid', 'RMSE_test']
    save_results_as_csv(indicators, columns_names, indicators_table_path)
    get_time_now()  # 获取当前时间
    print('Finished.')


if __name__ == '__main__':
    main()
