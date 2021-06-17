# -*- encoding: utf-8 -*-
"""
@Modify Time    2021/5/26 9:09
@Author         Tunan
@Desciption
                核聚类

输入data
y_pred = Kernel_Kmeans(Data, k)
输出y_pred聚类后的标签
"""
import numpy as np
import random
import math
from sklearn import metrics


def Kernel(v1, v2):
    '''
    核函数，可对应修改成多维空间下的核函数
    :param v1: 二维空间坐标
    :param v2: 二维空间坐标
    :return:
    '''
    beta = 100
    x1 = v1[0]
    x2 = v1[1]
    y1 = v2[0]
    y2 = v2[1]
    Ker = np.exp(-((x1 - y1)**2 + (x2 - y2)**2)*beta)# 高斯核
    # Ker = x1*y1 + x2*y2#线性核

    return Ker


def kernel_distance(Data, centers):
    '''
    核空间下的距离度量
    :param Data: 一个数据点
    :param centers: 一个中心点
    :return:
    '''
    return math.sqrt(Kernel(Data, Data) - 2 * Kernel(Data, centers) + Kernel(centers, centers))


def recenter(cu):
    '''
    在每个类别中，分别以每个样本为类中心
    计算类内其它各样本点到类中心的距离，并算出距离之和
    距离之和为最小的类中心就是该类的类中心
    :param cu:一个簇所有数据
    :return: 簇内最佳中心点的索引
    '''
    DISTANCE = []
    for i in range(len(cu)):
        dis = 0
        for j in range(len(cu)):
            dis += kernel_distance(cu[i], cu[j])
        DISTANCE.append(dis)
    return np.argmin(DISTANCE)


def Kernel_Kmeans(Data, C):
    data_num = Data.shape[0]
    print('样本个数：', data_num)

    # 确定初始聚类中心
    centers = {}
    for index, i in enumerate(random.sample(range(Data.shape[0]), C)):
        centers[index] = Data[i]
    print('初始聚类中心：', centers)

    circle_num = 0
    MAX_circle = 20  # 最大迭代次数
    while circle_num < MAX_circle:
        print('第', circle_num, '次循环')
        circle_num += 1
        print('聚类中心：', centers)
        # 类内循环
        distance1 = []
        distance2 = []
        for i in range(C):
            for j in range(data_num):
                # 各样本点到聚类中心的聚类
                dH = kernel_distance(Data[j], centers[i])
                if i == 0:
                    distance1.append(dH)
                elif i == 1:
                    distance2.append(dH)

        # 归属度矩阵，0表示第1类，1表示第2类
        y_pred = []
        for j in range(data_num):
            if distance1[j] > distance2[j]:
                y_pred.append(0)
            else:
                y_pred.append(1)

        #获取每一类的索引，可简化
        C1 = []
        C2 = []
        for i in range(C):
            for idx, value in enumerate(y_pred):
                # 获取第一类索引
                if value == 0:
                    C1.append(idx)
                elif value == 1:
                    C2.append(idx)

        #进行分簇，为简化思路设置的，可省略
        cu1 = []
        cu2 = []
        for i in range(data_num):
            if i in C1:
                cu1.append(Data[i])
            else:
                cu2.append(Data[i])

        # 计算误差
        score = metrics.calinski_harabasz_score(Data, y_pred)
        print('当前聚类结果的CH指数为', score)

        #重新规定质心
        centers = {}
        center1 = recenter(cu1)
        center2 = recenter(cu2)
        centers[0] = cu1[center1]
        centers[1] = cu2[center2]

    return y_pred


