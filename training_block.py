from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import torch
from model_params import *
from torch.nn.functional import mse_loss

torch.set_printoptions(threshold=5000)  # 设置一个足够大的阈值


# 获取AttentiveFP的预测值
def AFP_predict(model, data):
    # 将模型设置为评估模式
    model.eval()
    # 初始化一个空列表，用于存储预测值
    y_pred_list = []
    # 对每个批次进行循环
    for index, data_i in enumerate(data):
        # 将数据移动到设备上
        data_i = data_i.to(devices)
        # 前向传播，得到预测值
        output = model(x=data_i.x, edge_index=data_i.edge_index, edge_attr=data_i.edge_feature, batch=data_i.batch)
        y_pred_list.append(output.detach().cpu().numpy())
    y_pred_list = list(np.concatenate(y_pred_list).reshape(-1))
    return y_pred_list


# 获取含指纹模型预测值
def predict(model, data):
    # 将模型设置为评估模式
    model.eval()
    # 初始化一个空列表，用于存储预测值
    y_pred_list = []
    # 对每个批次进行循环
    for index, data_i in enumerate(data):
        # 将数据移动到设备上
        data_i = data_i.to(devices)
        # 前向传播，得到预测值
        output = model(smiles_list=data_i.Smiles, x=data_i.x, edge_index=data_i.edge_index,
                       edge_attr=data_i.edge_feature, batch=data_i.batch)
        y_pred_list.append(output.detach().cpu().numpy())
    y_pred_list = list(np.concatenate(y_pred_list).reshape(-1))
    return y_pred_list


# 从图结构中获取真实pKa值
def get_true_value(data):
    y_true_list = []
    for data_i in data:
        data_i = data_i.to(devices)
        y_true_list.append(data_i.y.cpu().numpy())
    y_true_list = list(np.concatenate(y_true_list).reshape(-1))
    return y_true_list


# 训练模型
def train_FPGNN(model, data_train, _slice, optimizer, loss_function, devices):
    molecule_index = 0
    begin_index = 0
    i = 0
    loss = 0
    count = 0
    for index, data_train_i in enumerate(data_train):
        data_train_i = data_train_i.to(devices)
        # _slice.to(devices)
        y = data_train_i.y.unsqueeze(-1)
        # batch, begin_index, molecule_index = get_batch(_slice, data_train_i.x.shape[0], begin_index, molecule_index)
        # batch = batch.to(devices)
        model.train()
        output = model(smiles_list=data_train_i.Smiles, x=data_train_i.x, edge_index=data_train_i.edge_index,
                       edge_attr=data_train_i.edge_feature, batch=data_train_i.batch)
        loss_i = loss_function(output, y)
        optimizer.zero_grad()
        loss_i.backward()
        optimizer.step()
        i += 1
        loss += loss_i.item()
        count += 1
    return loss / count


# 定义一个评价函数，输入参数为模型，数据和设备
def evaluation_FPGNN(model, data, _slice, devices):
    # 将模型设置为评估模式
    model.eval()
    # 初始化一个空列表，用于存储真实值和预测值
    y_true = []
    y_pred = []
    # 对每个批次进行循环
    for index, data_i in enumerate(data):
        # 将数据移动到设备上
        data_i = data_i.to(devices)
        # 前向传播，得到预测值
        output = model(smiles_list=data_i.Smiles, x=data_i.x, edge_index=data_i.edge_index,
                       edge_attr=data_i.edge_feature, batch=data_i.batch)
        # 将真实值和预测值添加到列表中
        y_true.append(data_i.y.cpu().numpy())
        y_pred.append(output.detach().cpu().numpy())
    # 将列表转换为数组，方便计算
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    # 计算R2和RMSE指标
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # 返回指标值
    return r2, mae, rmse


def early_stopping(RMSE_valid_list, delta_RMSE_threshold=0.02, count=5, epoch_min_threshold=200, model=None,
                   save_path=None):
    RMSE_min = min(RMSE_valid_list)  # 当前最小的RMSE
    epoch = len(RMSE_valid_list)  # 当前已经进行的轮数
    if RMSE_valid_list[-1] == RMSE_min:  # 如果当前epoch的RMSE是最小值，则保存模型
        torch.save(model, save_path)
        print(f"Best model saved at {save_path}, epoch {epoch}.")

    if epoch <= epoch_min_threshold:  # 保证模型的训练轮数不低于epoch_min_threshold
        return False

    for i in range(1, count + 1):
        # 检查连续轮次的RMSE变化是否小于阈值
        if RMSE_valid_list[-(i + 1)] - RMSE_valid_list[-i] > delta_RMSE_threshold:
            return False  # 有一轮RMSE变化超过阈值，不触发早停

    # 连续轮次内的RMSE变化都小于阈值，触发早停
    print("Early stopping triggered.")

    return True


# 训练与评价，对模型进行训练，并计算其训练集、验证集、测试集和外部验证集指标
def train_and_eval(model, data_train, data_valid, data_test, data_ext, _slice_train, _slice_valid, _slice_test,
                   _slice_ext, optimizer, loss_function=mse_loss, epochs=500, devices=devices, eval_interval=1):
    # 训练epochs轮
    for epoch in range(epochs):
        R2_train_list, R2_valid_list, R2_test_list = [], [], []
        MAE_train_list, MAE_valid_list, MAE_test_list = [], [], []
        RMSE_train_list, RMSE_valid_list, RMSE_test_list = [], [], []
        loss_list = []
        R2_ext_2d_list, MAE_ext_2d_list, RMSE_ext_2d_list = [], [], []
        # 每一轮训练cv个模型
        for i in range(cv):
            loss = train_FPGNN(model[i], data_train[i], _slice_train[i], optimizer[i], loss_function, devices)
            R2_train, MAE_train, RMSE_train = evaluation_FPGNN(model[i], data_train[i], _slice_train[i], devices)
            R2_valid, MAE_valid, RMSE_valid = evaluation_FPGNN(model[i], data_valid[i], _slice_valid[i], devices)
            R2_test, MAE_test, RMSE_test = evaluation_FPGNN(model[i], data_test[i], _slice_test[i], devices)
            R2_ext_list, MAE_ext_list, RMSE_ext_list = [], [], []
            # 考虑到可能有多个外部验证集的情况，所以使用循环遍历所有外部验证集
            for j in range(len(data_ext)):
                R2_ext, MAE_ext, RMSE_ext = evaluation_FPGNN(model[i], data_ext[j], _slice_ext[j], devices)
                R2_ext_list.append(R2_ext)
                MAE_ext_list.append(MAE_ext)
                RMSE_ext_list.append(RMSE_ext)
            # 将各个指标添加到列表中
            loss_list.append(loss)
            R2_train_list.append(R2_train)
            R2_valid_list.append(R2_valid)
            R2_test_list.append(R2_test)
            MAE_train_list.append(MAE_train)
            MAE_valid_list.append(MAE_valid)
            MAE_test_list.append(MAE_test)
            RMSE_train_list.append(RMSE_train)
            RMSE_valid_list.append(RMSE_valid)
            RMSE_test_list.append(RMSE_test)
            R2_ext_2d_list.append(R2_ext_list)
            MAE_ext_2d_list.append(MAE_ext_list)
            RMSE_ext_2d_list.append(RMSE_ext_list)
            # 保存模型
            torch.save(model[i],
                       model_save_path.split('.')[0] + '_e_' + str(epoch + 1).zfill(3) + '_cv_' + str(i + 1).zfill(
                           2) + '.' + model_save_path.split('.')[1])
        # 计算训练、验证、测试和外部验证集的各个指标的均值和标准差
        loss_mean, loss_std = np.mean(loss_list), np.std(loss_list)
        R2_train_mean, R2_train_std = np.mean(R2_train_list), np.std(R2_train_list, ddof=1)
        R2_valid_mean, R2_valid_std = np.mean(R2_valid_list), np.std(R2_valid_list, ddof=1)
        R2_test_mean, R2_test_std = np.mean(R2_test_list), np.std(R2_test_list, ddof=1)
        R2_ext_mean_list, R2_ext_std_list = np.mean(R2_ext_2d_list, axis=0), np.std(R2_ext_2d_list, axis=0, ddof=1)
        MAE_train_mean, MAE_train_std = np.mean(MAE_train_list), np.std(MAE_train_list, ddof=1)
        MAE_valid_mean, MAE_valid_std = np.mean(MAE_valid_list), np.std(MAE_valid_list, ddof=1)
        MAE_test_mean, MAE_test_std = np.mean(MAE_test_list), np.std(MAE_test_list, ddof=1)
        MAE_ext_mean_list, MAE_ext_std_list = np.mean(MAE_ext_2d_list, axis=0), np.std(MAE_ext_2d_list, axis=0, ddof=1)
        RMSE_train_mean, RMSE_train_std = np.mean(RMSE_train_list), np.std(RMSE_train_list, ddof=1)
        RMSE_valid_mean, RMSE_valid_std = np.mean(RMSE_valid_list), np.std(RMSE_valid_list, ddof=1)
        RMSE_test_mean, RMSE_test_std = np.mean(RMSE_test_list), np.std(RMSE_test_list, ddof=1)
        RMSE_ext_mean_list, RMSE_ext_std_list = np.mean(RMSE_ext_2d_list, axis=0), np.std(RMSE_ext_2d_list, axis=0,
                                                                                          ddof=1)

        # 输出评价指标
        if (epoch + 1) % eval_interval == 0 or epoch + 1 == epochs:
            print('Epoch {}: loss: {:.4f} ± {:.4f}'.format(epoch + 1, loss_mean, loss_std))  # 打印每轮的训练损失
            print(
                'Epoch {}: R2_train:{:.4f} ± {:.4f}, MAE_train:{:.4f} ± {:.4f}, RMSE_train:{:.4f} ± {:.4f}'.format(
                    epoch + 1, R2_train_mean, R2_train_std, MAE_train_mean, MAE_train_std, RMSE_train_mean,
                    RMSE_train_std))
            print(
                'Epoch {}: R2_valid:{:.4f} ± {:.4f}, MAE_valid:{:.4f} ± {:.4f}, RMSE_valid:{:.4f} ± {:.4f}'.format(
                    epoch + 1, R2_valid_mean, R2_valid_std, MAE_valid_mean, MAE_valid_std, RMSE_valid_mean,
                    RMSE_valid_std))
            print('Epoch {}: R2_test:{:.4f} ± {:.4f}, MAE_test:{:.4f} ± {:.4f}, RMSE_test:{:.4f} ± {:.4f}'.format(
                epoch + 1, R2_test_mean, R2_test_std, MAE_test_mean, MAE_test_std, RMSE_test_mean, RMSE_test_std))

            for R2_ext_mean, R2_ext_std, MAE_ext_mean, MAE_ext_std, RMSE_ext_mean, RMSE_ext_std in zip(R2_ext_mean_list,
                                                                                                       R2_ext_std_list,
                                                                                                       MAE_ext_mean_list,
                                                                                                       MAE_ext_std_list,
                                                                                                       RMSE_ext_mean_list,
                                                                                                       RMSE_ext_std_list):
                print('Epoch {}: R2_ext:{:.4f} ± {:.4f}, MAE_ext:{:.4f} ± {:.4f}, RMSE_ext:{:.4f} ± {:.4f}'.format(
                    epoch + 1, R2_ext_mean, R2_ext_std, MAE_ext_mean, MAE_ext_std, RMSE_ext_mean, RMSE_ext_std))
