### README

​		本次模型使用`3_train_classification.csv`文件作为训练集，`classification_test.csv`作为测试集，train文件夹中包含了其他的数据集，也可以将它们当作测试集或训练集用。

​		`classification.ipynb`和`classification.py`代码内容相同，是模型的训练代码；

​		`test_interface_classification.ipynb`是测试接口，用来测试模型误差。

​		将train中各个数据集当作测试集，测试模型误差，所得结果可以整理成下表：

|        样本         | sigmoid（Accruacy） | relu（Accruacy） |
| :-----------------: | :-----------------: | :--------------: |
|      训练样本       |      0.993750       |     1.000000     |
| 1_train_classification  |      0.987500       |     0.968750     |
| 2_train_classification  |      0.975000       |     0.956250     |
| 3_train_classification  |      0.975000       |     0.975000     |
| 4_train_classification  |      0.975000       |     0.968750     |
| 5_train_classification  |      0.975000       |     0.956250     |
| 6_train_classification  |      0.975000       |     0.962500     |
| 7_train_classification  |      0.975000       |     0.956250     |
| 8_train_classification  |      0.975000       |     0.956250     |
| 9_train_classification  |      0.975000       |     0.962500     |
| 10_train_classification |      0.993750       |     0.968750     |
| 11_train_classification |      0.968750       |     0.956250     |
| 12_train_classification |      0.975000       |     0.956250     |
| 13_train_classification |      0.975000       |     0.962500     |
| 14_train_classification |      0.981250       |     0.962500     |
| 15_train_classification |      0.981250       |     0.962500     |
| 16_train_classification |      0.975000       |     0.962500     |
| 17_train_classification |      0.975000       |     0.962500     |
| 18_train_classification |      0.975000       |     0.962500     |
| 19_train_classification |      0.962500       |     0.931250     |
| 20_train_classification |      0.975000       |     0.962500     |
| 21_train_classification |      0.981250       |     0.962500     |
| 22_train_classification |      0.987500       |     0.981250     |
