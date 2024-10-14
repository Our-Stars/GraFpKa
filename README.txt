GraFpKa
一个基于图神经网络的可解释性小分子pKa预测平台。

一、环境安装
项目中提供了环境文件env_GraFpKa.yml，直接执行以下命令即可完成环境安装：
conda env create -f env_GraFpKa.yml
之后，再激活环境即可：
conda activate env_GraFpKa

二、概览（Overview）
input\：输入数据，包括训练数据集、外部验证集和可电离位点文件。
model\：GraFpKa模型。
temp\：临时文件夹，用于存储临时数据。
visualization\:可视化结果，用于存储可解释性分析的结果。

三、用法
1.如果您需要训练模型：
（1）可以执行以下命令进行预训练：
python pretraining.py
（2）再执行以下命令用于微调：
python fine_tuning.py
2.如果您只想使用模型：
（1）可以执行以下命令进行pKa值预测：
python predicting.py
（2）也可以执行一下命令进行可解释性分析：
python interpretability.py