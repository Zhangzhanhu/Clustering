# -*- encoding: utf-8 -*-
"""
@Modify Time    2021/5/26 9:09
@Author         Tunan
@Desciption
                AP聚类

"""
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle


def AP_clustering(data):
    '''

    :param data:
    :param labels_true:
    :return:
    '''
    center_num=[]
    for i in range(-20,-50,-5):
        ap = AffinityPropagation(preference=i).fit(data)
        cluster_centers_indices = ap.cluster_centers_indices_    # 预测出的中心点的索引，如[123,23,34]
        labels = ap.labels_    # 预测出的每个数据的类别标签,labels是一个NumPy数组
        n_clusters_ = len(cluster_centers_indices)    # 预测聚类中心的个数
        center_num.append(n_clusters_)

    print('预测的聚类中心个数：%d' % n_clusters_)

    # 绘图
    plt.figure(1)    # 产生一个新的图形
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # 循环为每个类标记不同的颜色
    for k, col in zip(range(n_clusters_), colors):
        # labels == k 使用k与labels数组中的每个值进行比较
        # 如labels = [1,0],k=0,则‘labels==k’的结果为[False, True]
        class_members = labels == k
        cluster_center = data[cluster_centers_indices[k]]    # 聚类中心的坐标
        plt.plot(data[class_members, 0], data[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in data[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('clustering result')
    plt.show()

    return labels