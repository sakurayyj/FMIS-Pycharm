import numpy as np
import matplotlib.pyplot as plt
from numpy import mat
from sklearn.linear_model import Lasso
import pandas as pd
from collections import Counter
import csv
import warnings
import seaborn as sns
import matplotlib
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
from kmeans import *
from mysql import MySQL
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

def Pretrearment():

    '''
    x1	社会从业人数
    x2	在岗职工工资总额
    x3	社会消费品零售总额
    x4	城镇居民人均可支配收入
    x5	城镇居民人均消费性支出
    x6	年末总人口
    x7	全社会固定资产投资额
    x8	地区生产总值
    x9	第一产业产值
    x10	税收
    x11	居民消费价格指数
    x12	第三产业与第二产业产值比
    x13	居民消费水平
    y	财政收入
    '''
    # data = results
    # print(results)
    MySQL()
    data = pd.read_csv('code_data/data1.csv')
    x_data = data.drop('y', 1)
    y_data = data.loc[:, 'y']
    name_data = list(data.columns.values)
    # x_data = data.drop(index=0)
    # y_data = data.drop(columns=['y', 'year'])
    # x_data = data.drop(columns=['year','y'])  # 将y列和year列剔除
    # y_data = data.loc[:, 'y']  # 选取财政收入这一列的数据
    # print(x_data)
    # print(y_data)
    # name_data = list(data.columns.values)
    # print(name_data)

# vvv
    warnings.filterwarnings("ignore")  # 排除警告信息
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig.set_size_inches(14, 9)
    for i in range(13):
        ax = fig.add_subplot(4, 4, i + 1)
        x = x_data.loc[:, name_data[i]]
        y = y_data
        plt.title(name_data[i])
        ax.scatter(x, y)
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig("static/IMG/dataImg/examples.jpg")
    # plt.show()

    # #箱型图
    # sns.boxplot(data=x_data.loc[:, ['x1']])
    # plt.savefig("static/IMG/dataImg/examples1.jpg")
    # plt.show()


    # plt.figure(figsize=(7, 5), dpi=700)
    # plt.rcParams['figure.figsize'] = (8.0, 4.0)  # 设置figure_size尺寸
    # # plt.rcParams['figure.figsize'] = (5.5, 5)
    # plt.rcParams['savefig.dpi'] = 700  # 图片像素
    # plt.rcParams['figure.dpi'] = 700  # 分辨率
    plt.figure(figsize=(12, 12))
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='PuBu')
    plt.title('各个特征中的相关性')
    plt.savefig("static/IMG/dataImg/examples2.jpg",dpi = 200)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/assets/examples2.jpg",dpi = 200)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/res/drawable/examples2.jpg",dpi = 200)
    # plt.show()

    print(data.corr()['y'].sort_values())

    # 多变量的研究
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    sns.pairplot(data[["x1", "x2", "x3", "x4"]])
    plt.savefig("static/IMG/dataImg/examples3.jpg",dpi = 200)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/assets/examples3.jpg", dpi=200)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/res/drawable/examples3.jpg", dpi=200)
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    sns.pairplot(data[["x5", "x6", "x7", "x8"]])
    plt.savefig("static/IMG/dataImg/examples4.jpg",dpi = 200)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/assets/examples4.jpg", dpi=200)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/res/drawable/examples4.jpg", dpi=200)
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    sns.pairplot(data[["x9", "x10", "x12", "x13"]])
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig("static/IMG/dataImg/examples5.jpg",dpi = 200)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/assets/examples5.jpg", dpi=200)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/res/drawable/examples5.jpg", dpi=200)
    # plt.show()

    # 选取 x3,x5,x7特征数据
    x = x_data.drop('x11', 1)
    name = list(x.columns.values)
    for i in name:
        if i == 'x3':
            continue
        elif i == 'x5':
            continue
        elif i == 'x7':
            continue
        else:
            x = x.drop(i, 1)

    print(x)
    # 数据预处理
    train_data, test_data, train_target, test_target = train_test_split(x, y_data, test_size=0.3)
    Stand_X = StandardScaler()  # 把特征进行标准化
    Stand_Y = StandardScaler()  # 标签也是数值，也需要进行标准化
    train_data = Stand_X.fit_transform(train_data)
    test_data = Stand_X.transform(test_data)
    train_target = Stand_Y.fit_transform(train_target.values.reshape(-1, 1))  # reshape(-1,1)指将它转化为1列，行自动确定
    test_target = Stand_Y.transform(test_target.values.reshape(-1, 1))

    print(train_data)
    print(train_target)
    clf = DecisionTreeRegressor()
    clf.fit(train_data, train_target)
    y_pred = clf.predict(test_data)
    print("线性核函数：")
    print("训练集评分：", clf.score(train_data, train_target))
    print("测试集评分：", clf.score(test_data, test_target))

    # 绘制预测值和真实值对比图
    hos_pre = pd.DataFrame()
    hos_pre['predict'] = y_pred
    hos_pre['actual'] = test_target
    hos_pre.plot(figsize=(12, 8))
    plt.title("地方财政数据收入回归预测结果图")
    plt.savefig("static/IMG/dataImg/examples6.jpg",dpi = 150)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/assets/examples6.jpg", dpi=150)
    plt.savefig("D:/AndroidStudioTest/FMIS/app/src/main/res/drawable/examples6.jpg", dpi=150)
    # plt.show()

    MSE = np.sum((y_pred - test_target) ** 2) / len(test_target)
    MAE = np.sum(np.absolute(y_pred - test_target)) / len(test_target)
    print("测试集均方差：", metrics.mean_squared_error(test_target, y_pred.reshape(-1, 1)))
    print("测试集R2分：", metrics.r2_score(test_target, y_pred.reshape(-1, 1)))
    print("MSE", MSE)
    print("MAE", MAE)
    print('RMSE:{:.4f}'.format(sqrt(MSE)))  # RMSE(标准误差)

    cluster_Num = 3
    data = mat(train_data)
    centroids, clusterAssment = KMeans(data, cluster_Num)
    showCluster(data, cluster_Num, clusterAssment, centroids)