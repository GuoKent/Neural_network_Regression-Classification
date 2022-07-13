import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
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
class regression(nn.Module):
    def __init__(self):
        # 初始化nn.Module
        super(regression, self).__init__()

        # 添加隐藏层
        # 第一层隐藏层
        self.hid1 = nn.Linear(13, 26)  # 13为输入参数的个数，即X1-X13，后者为隐藏层神经元个数
        # 第二层隐藏层
        self.hid2 = nn.Linear(26, 26)  # 前者为上一层隐藏层神经元个数，后者为该层神经元个数
        # 第三层隐藏层
        self.hid3 = nn.Linear(26, 26)  # 前者为上一层隐藏层神经元个数，后者为该层神经元个数
        # 隐藏层输出y
        self.f = nn.Linear(26, 1)  # 前者为最后一层隐藏层神经元个数，后者为输出值y，y只有一个特征

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
        x = self.hid3(x)
        nn.Dropout(0.5),  # 以0.5的概率断开
        x = self.sigmoid(x)  # 第二层神经元的激活函数
        x = self.f(x)
        x = x.squeeze(-1)  # 降1维，否则后面输入维度不同
        return x


if __name__ == '__main__':
    X, Y = load_data('3_train_regression.csv')
    X = np.array(X)
    Y = np.array(Y)
    loss_train = 0
    loss_test = 0
    MAPE_train = 0
    MAPE_test = 0
    # 模型
    model = regression()
    # 采用均方根误差MSE作为代价函数
    mse = nn.MSELoss()
    # 优化器，包含各参数w和θ和学习步长
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 数据预处理，归一化
    mm = MinMaxScaler()
    X = mm.fit_transform(X)
    # 把numpy的数据转化成tensor
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)

    # 训练模型
    inputs = Variable(X)
    target = Variable(Y)
    epoch = 1500001  # 迭代次数
    for i in range(epoch):
        out = model.forward(inputs)  # out=model(inputs)?
        loss = mse(out, target)
        # 清空过往梯度，若不清空，梯度会累加
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 根据新梯度修改权值
        optimizer.step()

    # 5次5折交叉验证
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # 使用训练集训练模型
        inputs = Variable(X_train)
        target = Variable(Y_train)
        y_train = model.forward(inputs)  # y为最终训练所得模型算出来的预测值
        loss_train += mse(y_train, target)
        y_train = y_train.detach().numpy()  # 转化为numpy计算MAPE
        Y_train = Y_train.detach().numpy()
        MAPE_train += np.mean(np.abs((y_train - Y_train) / Y_train)) * 100

        inputs = Variable(X_test)
        target = Variable(Y_test)
        y_test = model.forward(inputs)  # y为最终训练所得模型算出来的预测值
        loss_test += mse(y_test, target)
        y_test = y_test.detach().numpy()  # 转化为numpy计算MAPE
        Y_test = Y_test.detach().numpy()
        MAPE_test += np.mean(np.abs((y_test - Y_test) / Y_test)) * 100

    torch.save(model.state_dict(), 'model_regression.pkl')
    print('模型评估结果:')
    print('迭代次数:%d' % (epoch-1))
    print('平均训练误差(MSE):')
    print(loss_train.item() / 25)
    print('平均训练误差(MAPE):')
    print(MAPE_train / 25, end='%\n')
    print('平均测试误差(MSE):')
    print(loss_test.item() / 25)
    print('平均测试误差(MAPE):')
    print(MAPE_test / 25, end='%\n')

