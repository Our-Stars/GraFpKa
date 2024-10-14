import matplotlib.pyplot as plt
from rdkit import Chem
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image, ImageOps
from matplotlib.cm import ScalarMappable
from model_params import devices
from matplotlib.colors import Normalize
import torch
from captum.attr import IntegratedGradients


# 以下的8个函数均用于根据权重绘制不同原子或不同键的高亮颜色，用于权重可视化
# 将权重转化为rgb三元组
def weight_to_rgb(weight, min_weight, max_weight):
    # Normalize weight to [0, 1]
    normalized_weight = (weight - min_weight) / (max_weight - min_weight)
    # Map weight to RGB using a colormap
    rgba_color = plt.get_cmap('bwr')(normalized_weight)
    # Convert to RGB
    rgb_color = (rgba_color[0], rgba_color[1], rgba_color[2])
    return rgb_color


# 添加色带
def add_colorbar(ax, cmap, norm, label):
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need to set an array for the ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
    cbar.set_label(label, rotation=270, labelpad=15)


# 以下四个函数来自http://rdkit.chenzhaoqiang.com/mediaManual.html#d2d-drawmolecule的高亮原子和键的颜色 d2d.DrawMolecule
def _drawerToImage(d2d):
    try:
        import Image
    except ImportError:
        from PIL import Image
    sio = BytesIO(d2d.GetDrawingText())
    return Image.open(sio)


def clourMol(mol, highlightAtoms_p=None, highlightAtomColors_p=None, highlightBonds_p=None, highlightBondColors_p=None,
             sz=[1000, 1000]):
    '''

    '''
    d2d = rdMolDraw2D.MolDraw2DCairo(sz[0], sz[1])
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 20
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p, highlightAtomColors=highlightAtomColors_p,
                     highlightBonds=highlightBonds_p, highlightBondColors=highlightBondColors_p)
    d2d.FinishDrawing()
    product_img = _drawerToImage(d2d)
    return product_img


def StripAlphaFromImage(img):
    '''This function takes an RGBA PIL image and returns an RGB image'''

    if len(img.split()) == 3:
        return img
    return Image.merge('RGB', img.split()[:3])


def TrimImgByWhite(img, padding=10):
    '''This function takes a PIL image, img, and crops it to the minimum rectangle
    based on its whiteness/transparency. 5 pixel padding used automatically.'''

    # Convert to array
    as_array = np.array(img)  # N x N x (r,g,b,a)

    # Set previously-transparent pixels to white
    if as_array.shape[2] == 4:
        as_array[as_array[:, :, 3] == 0] = [255, 255, 255, 0]

    as_array = as_array[:, :, :3]

    # Content defined as non-white and non-transparent pixel
    has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
    xs, ys = np.nonzero(has_content)

    # Crop down
    margin = 5
    x_range = max([min(xs) - margin, 0]), min([max(xs) + margin, as_array.shape[0]])
    y_range = max([min(ys) - margin, 0]), min([max(ys) + margin, as_array.shape[1]])
    as_array_cropped = as_array[
                       x_range[0]:x_range[1], y_range[0]:y_range[1], 0:3]

    img = Image.fromarray(as_array_cropped, mode='RGB')

    return ImageOps.expand(img, border=padding, fill=(255, 255, 255))


# 根据权重高亮一个分子中的原子或键，权重越大，颜色越红，权重越小，颜色越蓝(如果只有原子有权重，则只能可视化原子，如AttentiveFP，如果只有键有权重，则只能可视化键，如GAT，如果都没有则不能可视化，如GCN)
def highlight_atoms_or_bonds_from_weights(smiles, index, weights, save_path, atoms_or_bonds, normalize,
                                          threshold=-1):
    mol = Chem.MolFromSmiles(smiles)
    min_weight = min(weights)
    max_weight = max(weights)
    if threshold < 0:
        threshold = max(abs(min_weight), abs(max_weight))
    if normalize:  # 是否进行标准化，基于权重的可视化需要标准化，基于积分梯度的可视化不需要标准化
        highlightColors_p = {atom: weight_to_rgb(weight, min_weight, max_weight) for atom, weight in
                             zip(index, weights)}  # 计算权重对应的RGB
    else:
        highlightColors_p = {atom: weight_to_rgb(weight, -threshold, threshold) for atom, weight in
                             zip(index, weights)}  # 计算权重对应的RGB
    fig, ax = plt.subplots()
    # 以下代码来自“http://rdkit.chenzhaoqiang.com/mediaManual.html#d2d-drawmolecule的高亮原子和键的颜色 d2d.DrawMolecule”部分的示例代码
    if atoms_or_bonds == 'atoms':
        img1 = clourMol(mol, highlightAtoms_p=index, highlightAtomColors_p=highlightColors_p)
    elif atoms_or_bonds == 'bonds':
        img1 = clourMol(mol, highlightBonds_p=index, highlightBondColors_p=highlightColors_p)
    img1 = StripAlphaFromImage(img1)
    img1 = TrimImgByWhite(img1)
    ax.imshow(np.asarray(img1))
    ax.axis('off')
    # 添加颜色条
    if normalize:
        norm = Normalize(vmin=0, vmax=1)
    else:
        norm = Normalize(vmin=-threshold, vmax=+threshold)
    add_colorbar(ax, plt.get_cmap('bwr'), norm, 'IG Values')
    # 调整布局，确保颜色条不会出现在图片外
    plt.subplots_adjust(right=0.85)
    # 保存绘制结果
    plt.savefig(save_path, dpi=600, bbox_inches='tight')


# 根据权重批量高亮原子或键，权重越大，颜色越红，权重越小，颜色越蓝(如果只有原子有权重，则只能可视化原子，如AttentiveFP，如果只有键有权重，则只能可视化键，如GAT，如果都没有则不能可视化，如GCN)
def batch_highlight_atoms_or_bonds_from_weights(data, model, save_path, atoms_or_bonds, threshold):
    model.eval()
    for idx, data_i in enumerate(data):  # 处理所有分子，注意：要保证这里的data没有分批
        data_i = data_i.to(devices)
        mol = Chem.MolFromSmiles(data_i.Smiles)
        if atoms_or_bonds == 'atoms':  # 批量高亮原子
            atoms = mol.GetAtoms()  # 原子对象
            atoms_num = len([x for x in atoms])  # 原子个数
            output, w1, w2 = model(data_i)  # 计算输出
            index = w2[0][0, -atoms_num:]  # 获取需要高亮的原子索引
            index = list(index.cpu().numpy())  # 将张量转化为ndarray
            index = [int(atom) for atom in index]  # 将numpy.int64的数据转化为int
            weights = w2[1][-atoms_num:].view(-1)  # 获取原子权重
            weights = list(weights.detach().cpu().numpy())  # 将张量转化为ndarray
            final_path = save_path + '/' + atoms_or_bonds + '_' + str(idx) + '.png'
            highlight_atoms_or_bonds_from_weights(data_i.Smiles, index, weights, final_path, atoms_or_bonds, True,
                                                  threshold)  # 绘图
            plt.close()  # 关闭图形，避免占用过多内存
        elif atoms_or_bonds == 'bonds':  # 批量高亮键
            bonds = mol.GetBonds()  # 原子对象
            bonds_num = len([x for x in bonds])  # 键个数
            output, w1, w2 = model(data_i)  # 计算输出
            index = w2[0][:, :2 * bonds_num]  # 获取需要高亮的键索引
            weights = w2[1][:2 * bonds_num].view(-1)  # 获取键权重
            index_new = []
            # 根据键的起始原子索引和终止原子索引计算键的索引
            for i in range(len(index[0])):
                begin_index = index[0][i].item()  # 获取起始原子索引
                end_index = index[1][i].item()  # 获取终止原子索引
                bond = mol.GetBondBetweenAtoms(begin_index, end_index)  # 获取键对象
                index_new.append(bond.GetIdx())  # 获取键索引并添加到index_new列表中
            weights = list(weights.detach().cpu().numpy())  # 将张量转化为ndarray
            weights_new = []
            # 由于构造图的时候，使用了无向图，所以每条边有两个相同的索引和与之一一对应的两个权重，对于每一个键的两个权重，取其权重的平均值
            for i in range(bonds_num):  # 从0遍历到bonds_num-1，这是所有可能出现的键索引
                bond_index_list = []
                for j in range(len(index_new)):  # 遍历所有的键索引
                    if i == index_new[j]:  # 如果找到键索引分别为0,1,2，...，bonds_num-1的键，则将其位置保存下来
                        bond_index_list.append(j)  # bond_index_list长度一定是2，因为每个键出现了两次
                assert len(bond_index_list) == 2, '键连接关系有误！'
                weights_new.append((weights[bond_index_list[0]] + weights[bond_index_list[1]]) / 2)  # 求两个权重的均值
            final_path = save_path + '/' + atoms_or_bonds + '_' + str(idx) + '.png'
            highlight_atoms_or_bonds_from_weights(data_i.Smiles, index_new, weights_new, final_path,
                                                  atoms_or_bonds, True)  # 绘图
            plt.close()  # 关闭图形，避免占用过多内存


# 针对一个分子，根据输入计算积分梯度值（IG），将每一个原子的所有IG值求和作为该原子的权重，然后绘制权重热图
def highlight_atoms_or_bonds_from_IG(model, smiles, inputs, baselines, target, additional_forward_args,
                                     internal_batch_size, save_path, devices, threshold=-1):
    '''
    以下注释来自：captum.attr.IntegratedGradients类：
            inputs (Tensor or tuple[Tensor, ...]): Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define the starting point from which integral
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            internal_batch_size (int, optional): Divides total #steps * #examples
                        data points into chunks of size at most internal_batch_size,
                        which are computed (forward / backward passes)
                        sequentially. internal_batch_size must be at least equal to
                        #examples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain internal_batch_size / num_devices examples.
                        If internal_batch_size is None, then all evaluations are
                        processed in one batch.
                        Default: None
                        '''
    model = model.to(devices)
    ig = IntegratedGradients(model)  # 定义IntegratedGradients对象
    attributions = ig.attribute(inputs=inputs, baselines=baselines, additional_forward_args=additional_forward_args,
                                target=target, internal_batch_size=internal_batch_size)  # 计算归因值（即积分梯度值）
    # print(attributions)
    weight = attributions.sum(dim=1).cpu()  # 计算权重，以每一个原子的所有特征的积分梯度值之和作为权重
    index = [x for x in range(inputs.shape[0])]
    atoms_or_bonds = 'atoms'
    highlight_atoms_or_bonds_from_weights(smiles=smiles, index=index, weights=weight, save_path=save_path,
                                          atoms_or_bonds=atoms_or_bonds, normalize=False, threshold=threshold)  # 可视化


# 针对多个分子，根据输入计算积分梯度值（IG），将每一个原子的所有IG值求和作为该原子的权重，然后绘制权重热图，注意：这里的data必须是DataLoader分批前的数据
def batch_highlight_atoms_or_bonds_from_IG(data, model, save_path, devices, threshold=-1):
    target = None
    for i, data_i in enumerate(data):
        data_i.to(devices)
        baselines = torch.zeros(data_i.x.shape).to(devices)
        batch = torch.zeros(data_i.x.shape[0], dtype=torch.int64).to(devices)
        internal_batch_size = data_i.edge_index.shape[1] / 2
        save_path_i = save_path + str(i).zfill(4) + '.png'  # 需要更多数量时，可以把4改成更大的数字
        highlight_atoms_or_bonds_from_IG(model=model,
                                         smiles=data_i.Smiles,
                                         inputs=data_i.x,
                                         baselines=baselines,
                                         target=target,
                                         additional_forward_args=(
                                             data_i.edge_index, data_i.edge_feature, batch, [data_i.Smiles]),
                                         internal_batch_size=internal_batch_size,
                                         save_path=save_path_i,
                                         devices=devices,
                                         threshold=threshold
                                         )
