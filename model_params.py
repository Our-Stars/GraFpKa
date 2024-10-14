from torch.cuda import is_available
from model import GCN, AttentiveFP, GAT, FP_GNN

# 数据预处理参数
mw_low = 100
mw_high = 799

# 超参数
batch_size = 512  # 批大小
epochs = 100  # 最大训练轮数
lr = 5e-5  # 学习率(人工调参：GCN和AttentiveFP:0.005,GAT:0.002)(微调：酸性：3e-4或1e-3，碱性：5e-6或1e-5)
weight_decay = 5e-4  # 权重衰减系数(5e-4)
delta_RMSE_threshold = 0.02  # 连续count轮模型性能提升低于delta_RMSE_threshold则早停,数字越大越容易早停，建议设置0.01~0.03
count = 8  # 连续count轮模型性能提升低于threshold则早停，数字越大越难早停，建议设置5~10
epoch_min_threshold = 4  # 至少要训练模型epoch_min_threshold轮才结束，至少设为1
devices = "cuda" if is_available() else "cpu"  # 设备cpu或gpu

# 模型列表及超参数字典
gcn_params = {  # GCN的超参数
    "atom_feature_dim": 49,
    "hidden_channels": 256,
    "out_channels": 1,
    "dropout": 0.1
}
afp_params = {  # AttentiveFP的超参数
    "in_channels": 49,
    "hidden_channels": 128,
    "out_channels": 1,
    "edge_dim": 12,
    "num_layers": 2,
    "num_timesteps": 2,
    "dropout": 0.3
}
gat_params = {  # GAT的超参数
    "atom_feature_dim": 49,
    "hidden_channels": 256,  # 64
    "out_channels": 1,
    "head_num": 1,
    "dropout": 0.1  # 0
}
fpgnn_params = {  # FP_GNN的超参数
    "atom_feature_dim": 49,
    "edge_dim": 12,
    "hidden_channels_AFP": 128,
    "out_channels_AFP": 128,
    "hidden_channels_FPN": 128,
    "out_channels_FPN": 128,
    "hidden_channels_FFN": 128,
    "fp_type": 'mixed'
}
model_dict = {  # 模型字典
    "GCN": {"class": GCN, "params": gcn_params},
    "AttentiveFP": {"class": AttentiveFP, "params": afp_params},
    "GAT": {"class": GAT, "params": gat_params},
    "FPGNN": {"class": FP_GNN, "params": fpgnn_params}
}

# 其他参数
model_name = 'FPGNN'  # 当前需要训练的模型名称(GCN或AttentiveFP或GAT或FPGNN)
acid_or_base = 'base'  # 当前是酸性数据或是碱性数据，可以填写“acid”或者“base”或者None
macro_or_micro = 'macro'  # 直接预测宏观pKa，还是通过微观pKa预测宏观pKa，可以填写“macro”或者“micro”
cal_ionization_center_bool_list = True  # 是否计算电离中心布尔列表，如果计算，设为True，否则设为False
dataset_name = 'SAMPL6'  # 当前外部验证集的名称，可以选择“SAMPL6”或“novartis”
cv = 5  # 随机划分训练集、验证集和测试集的次数

# 绘图时权重的最大最小值
min_weight = 0
max_weight = 1

# 路径
result_num = 35  # 第几次的结果
csv_file_path = 'input/DataWarrior.csv'  # 原始csv文件路径
smarts_file_path = 'input/smarts_pattern_ionized.txt'  # smarts文件路径
file_path_preprocessed = 'input/Smiles_basicOrAcidic_pKa_preprocessed.csv'  # 预处理后的文件保存位置
acid_save_path = 'input/Smiles_basicOrAcidic_pKa_acid.csv'  # 酸碱分离后，酸性物质的保存路径
base_save_path = 'input/Smiles_basicOrAcidic_pKa_base.csv'  # 酸碱分离后，碱性物质的保存路径
# 数据集划分后保存路径
acid_train_save_path = 'division_results/division_results_' + str(result_num) + '/acid_train_data.csv'
acid_valid_save_path = 'division_results/division_results_' + str(result_num) + '/acid_valid_data.csv'
acid_test_save_path = 'division_results/division_results_' + str(result_num) + '/acid_test_data.csv'
base_train_save_path = 'division_results/division_results_' + str(result_num) + '/base_train_data.csv'
base_valid_save_path = 'division_results/division_results_' + str(result_num) + '/base_valid_data.csv'
base_test_save_path = 'division_results/division_results_' + str(result_num) + '/base_test_data.csv'
# 根据当前处理的酸碱数据的不同，路径有所不同
if acid_or_base == 'acid' or acid_or_base == 'base':  # 处理酸性或碱性的情况
    tsne_path = 'results/results_' + str(result_num) + '/' + acid_or_base + '/t-SNE'  # t-SNE图像保存路径
    plot_base_path = 'results/results_' + str(result_num) + '/' + acid_or_base + '/Figure_' + str(
        result_num) + '_'  # 图像文件保存路径
    indicators_table_path = 'results/results_' + str(
        result_num) + '/' + acid_or_base + '/indicators_table.csv'  # 每一轮的指标文件保存路径
    model_save_path = 'model/' + 'model_' + str(
        result_num) + '/' + model_name + '_' + acid_or_base + '_' + macro_or_micro + '_finetuning_model.pth'  # 模型保存路径
    true_pred_scatter_save_path = 'results/results_' + str(
        result_num) + '/' + acid_or_base + '/true_pred_scatter'  # 真实值-预测值散点图保存路径
elif acid_or_base == None:  # 不分离酸碱的情况
    tsne_path = 'results/results_' + str(result_num) + '/t-SNE'  # t-SNE图像保存路径
    plot_base_path = 'results/results_' + str(result_num) + '/Figure_' + str(result_num) + '_'  # 图像文件保存路径
    indicators_table_path = 'results/results_' + str(result_num) + '/indicators_table.csv'  # 每一轮的指标文件保存路径
    model_save_path = 'model/' + 'model_' + str(
        result_num) + '/' + model_name + '_' + acid_or_base + '_finetuning_model.pth'  # 模型保存路径
    true_pred_scatter_save_path = 'results/results_' + str(result_num) + '/true_pred_scatter'  # 真实值-预测值散点图保存路径

# 确定是否进行某些操作
if_preprocess = True  # 是否进行数据预处理
if_split_acid_base = True  # 是否进行酸碱分离
# 若if_split_train_test为True，则if_split_train_valid必须设置为True，否则数据划分会出现问题
if_split_train_test = True  # 是否划分训练集（包括训练集和验证集）和测试集
if_split_train_valid = True  # 是否划分训练集和验证集
