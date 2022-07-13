import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler


def load_data(filename):
    data = pd.read_csv(filename)  # 小数读取默认只保留6位
    data = np.array(data)
    Y = data[:, -1]
    X = np.delete(data, -1, axis=1)
    np.set_printoptions(suppress=True)  # 取消科学计数法表示
    return X, Y


# 构建神经网络
# 定义模型类
class classification(nn.Module):
    def __init__(self):
        # 初始化nn.Module
        super(classification, self).__init__()

        # 添加隐藏层
        # 第一层隐藏层
        self.hid1 = nn.Linear(13, 26)  # 13为输入参数的个数，即X1-X13，10为隐藏层神经元个数
        # 第二层隐藏层
        self.hid2 = nn.Linear(26, 26)  # 前者为上一层隐藏层神经元个数，后者为该层神经元个数
        # 隐藏层输出y
        self.f = nn.Linear(26, 3)  # 前者为最后一层隐藏层神经元个数，后者为输出值y，y分三类

        # 定义隐藏层的激活函数
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=False)

    #  定义神经网络的计算
    def forward(self, x):
        x = self.hid1(x)
        nn.Dropout(0.5),  # 以0.5的概率断开
        x = self.sigmoid(x)  # 第一层神经元的激活函数（每层一个激活函数）
        x = self.hid2(x)
        nn.Dropout(0.5),  # 以0.5的概率断开
        x = self.sigmoid(x)  # 第二层神经元的激活函数
        x = self.f(x)
        x = x.squeeze(-1)  # 降1维，否则后面输入维度不同
        return x


if __name__ == '__main__':
    X, Y = load_data('3_train_classification.csv')
    X = np.array(X)
    Y = np.array(Y)
    # 模型
    model = classification()
    # 数据预处理，归一化
    mm = MinMaxScaler()
    X = mm.fit_transform(X)
    # Softmax–Log–NLLLoss三合一的损失函数，用于多分类
    SFN = nn.CrossEntropyLoss()
    # 优化器，包含各参数w和θ和学习步长
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # 转化为tensor形式
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)

    # 训练模型
    inputs = X
    target = Y
    for i in range(20001):
        outputs = model.forward(inputs)
        # 计算损失值
        loss = SFN(outputs, target.long())
        # 清空过往梯度，若不清空，梯度会累加
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 根据新梯度修改权值
        optimizer.step()

    total = len(Y)
    correct = 0
    outputs = model.forward(inputs)
    predicted = torch.max(outputs.data, dim=1)  # 训练集的预测值
    for i in range(len(Y)):
        if Y[i] == predicted.indices[i]:
            correct += 1
    accuracy = correct / total
    print('模型训练正确个数:%d' % correct)
    print('模型训练错误个数:%d' % (total - correct))
    print('Accuracy=%f\n' % accuracy)

    # 五次五折交叉验证评估模型
    accuracy_train = 0
    accuracy_test = 0
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # 计算训练误差
        inputs = X_train
        target = Y_train
        correct_train = 0
        total = len(Y_train)
        outputs = model.forward(inputs)  # 训练集的输出结果
        predicted = torch.max(outputs.data, dim=1)  # 训练集的预测值
        # print(predicted)
        for i in range(len(Y_train)):
            if Y_train[i] == predicted.indices[i]:
                correct_train += 1
        accuracy_train += correct_train / total

        # 计算测试误差
        inputs = X_test
        target = Y_test
        correct_test = 0
        total = len(Y_test)
        outputs = model.forward(inputs)  # 测试集的输出结果
        predicted = torch.max(outputs.data, dim=1)  # 测试集的预测值
        for i in range(len(Y_test)):
            if Y_test[i] == predicted.indices[i]:
                correct_test += 1
        accuracy_test += correct_test / total

    torch.save(model.state_dict(), 'model_classification.pkl')
    print('模型评估结果:')
    print('平均训练Accuracy=%f' % (accuracy_train / 25))
    print('平均测试Accuracy=%f' % (accuracy_test / 25))
