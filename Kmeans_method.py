# -*- encoding: utf-8 -*-
"""
@Modify Time    2021/5/26 9:09
@Author         Tunan
@Desciption
对数据集使用肘部法确定K个数
使用K-means聚类

输入data
Ave_Distor(data)
y_pred = K_MEANS(data, 2)
输出y_pred为聚类后的标签

"""
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np


def K_MEANS(data,k):
    '''
    K-Means算法
    :param data: 输入数据  eg：[[7.94 -0.96],[-0.27 -7.61]...]
    :param k: 目标聚类簇数
    :return: y_pred聚类后的标签
    '''
    score = []
    for i in range(1, 100):
        y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(data)
        defen = metrics.calinski_harabasz_score(data,y_pred)
        score.append(defen)
    plt.plot(score)
    plt.show()

    return y_pred


def Ave_Distor(data):
    '''
    肘部法确定K值
    :param data: 输入数据  eg：[[7.94 -0.96],[-0.27 -7.61]...]
    :return: None
    '''
    K = range(1,10)
    meandistortions = []
    for k in K:
        Kmeans = KMeans(n_clusters=k)
        Kmeans.fit(data)
        meandistortions.append(sum(np.min(cdist(data,Kmeans.cluster_centers_,'euclidean'), axis=1))/data.shape[0])
    plt.plot(K, meandistortions,'bx-')
    plt.xlabel('K')
    plt.ylabel('Ave Distor')
    plt.show()

