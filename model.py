# 导入必要的库和模块
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn.models import AttentiveFP
from joblib import Parallel, delayed
import torch
from rdkit import Chem
from utils import Mol_Des_Fp


# 定义一个GCN类，继承自torch.nn.Module类，实现__init__方法和forward方法。
class GCN(nn.Module):
    def __init__(self, atom_feature_dim, hidden_channels, out_channels, dropout):
        super(GCN, self).__init__()
        # 初始化两个图卷积层，分别用于输入层和隐藏层。
        self.conv1 = GCNConv(atom_feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # 初始化一个全连接层，用于输出层。
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # 输入图数据对象，包含节点特征（x），边索引（edge_index）。
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GAT(nn.Module):
    def __init__(self, atom_feature_dim, hidden_channels, out_channels, head_num=1, dropout=0):
        super(GAT, self).__init__()
        self.GAT1 = GATConv(in_channels=atom_feature_dim, out_channels=hidden_channels, heads=head_num, dropout=dropout)
        self.GAT2 = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=head_num, dropout=dropout)
        self.ln = nn.LayerNorm(hidden_channels)
        self.fc1 = nn.Linear(in_features=hidden_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_attr, batch, return_attention_weights=True):
        # return_attention_weights只能选择True或None，不能选择False，True表示返回权重，None表示不返回权重
        if return_attention_weights:
            x, w1 = self.GAT1(x=x, edge_index=edge_index, edge_attr=edge_attr,
                              return_attention_weights=return_attention_weights)
            x = nn.LeakyReLU(0.1)(self.ln(x))
            x, w2 = self.GAT2(x=x, edge_index=edge_index, edge_attr=edge_attr,
                              return_attention_weights=return_attention_weights)
            x = nn.LeakyReLU(0.1)(self.ln(x))
            x = global_max_pool(x, batch)
            x = F.relu(x)
            x = self.fc1(x)
            return x.view(-1), w1, w2
        else:
            x = self.GAT1(x=x, edge_index=edge_index, edge_attr=edge_attr,
                          return_attention_weights=return_attention_weights)
            x = nn.LeakyReLU(0.1)(self.ln(x))
            x = self.GAT2(x=x, edge_index=edge_index, edge_attr=edge_attr,
                          return_attention_weights=return_attention_weights)
            x = nn.LeakyReLU(0.1)(self.ln(x))
            x = global_max_pool(x, batch)
            x = F.relu(x)
            x = self.fc1(x)
            return x.view(-1)




class FPN(nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0.0, fp_type='mixed'):
        super(FPN, self).__init__()
        self.fp_type = fp_type
        in_channels = 1363 if self.fp_type == 'mixed' else 1024
        self.fc1 = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        self.fc2 = nn.Linear(in_features=hidden_channels, out_features=out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def compute_fp(self, smiles):
        fp = []
        mol = Chem.MolFromSmiles(smiles)
        if self.fp_type == 'mixed':
            mol_fp = Mol_Des_Fp(mol)
            fp_maccs = mol_fp.get_maccs_fp()
            fp_pubchem = mol_fp.get_pubchem_fp()
            fp_phaErG = mol_fp.get_erg_fp()
            fp.extend(fp_maccs)
            fp.extend(fp_pubchem)
            fp.extend(fp_phaErG)
        elif self.fp_type == 'morgan':
            mol_fp = Mol_Des_Fp(mol)
            fp_morgan = mol_fp.get_morgan_fp(radius=2, nBits=1024)
            fp = fp_morgan
        return fp

    def forward(self, smiles_list):
        # Compute fingerprints in parallel
        fps_list = Parallel(n_jobs=-1)(delayed(self.compute_fp)(smiles) for smiles in smiles_list)

        # Convert to CUDA tensor
        fps_tensor = torch.tensor(fps_list).to(next(self.parameters()).device)

        # Pass through neural network layers
        x = self.fc1(fps_tensor)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class FP_GNN(nn.Module):
    def __init__(self, atom_feature_dim, edge_dim, hidden_channels_AFP, out_channels_AFP, hidden_channels_FPN,
                 out_channels_FPN, hidden_channels_FFN, num_layers_AFP=2, num_timesteps_AFP=2, dropout_AFP=0.0,
                 dropout_FPN=0.0, dropout_FFN=0.0, fp_type='mixed'):
        super(FP_GNN, self).__init__()
        self.AFP = AttentiveFP(in_channels=atom_feature_dim, hidden_channels=hidden_channels_AFP,
                               out_channels=out_channels_AFP, edge_dim=edge_dim, num_layers=num_layers_AFP,
                               num_timesteps=num_timesteps_AFP, dropout=dropout_AFP)
        self.FPN = FPN(hidden_channels=hidden_channels_FPN, out_channels=out_channels_FPN, dropout=dropout_FPN,
                       fp_type=fp_type)
        self.FFN = nn.Sequential(nn.Dropout(p=dropout_FFN),
                                 nn.Linear(in_features=out_channels_AFP + out_channels_FPN,
                                           out_features=hidden_channels_FFN),
                                 nn.ReLU(),
                                 nn.Dropout(p=dropout_FFN),
                                 nn.Linear(in_features=hidden_channels_FFN, out_features=1))

    def forward(self, x, edge_index, edge_attr, batch, smiles_list):
        x_AFP = self.AFP(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x_FPN = self.FPN(smiles_list=smiles_list)
        x = torch.cat([x_AFP, x_FPN], axis=1)
        x = self.FFN(x)
        return x