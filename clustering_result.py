# -*- encoding: utf-8 -*-
"""
@Modify Time    2021/6/17 14:33  
@Author         Tunan
@Filename       clustering_result.py
@Desciption     API类型调用聚类方法，选择显示聚类结果
                  
"""
from sklearn import metrics
from sklearn.datasets import make_blobs
import Kmeans_method
import kernel_method
import AP_clustering
import spectral_clustering

def load_data():
    X, y = make_blobs(n_samples=300, centers=[[1,1],[-1,-1]],n_features=3)

    return X, y

def cluster(X, method):
    global y_pred
    if method == 'KMeans':
        Kmeans_method.Ave_Distor(X)
        y_pred = Kmeans_method.K_MEANS(X,k=2)
    elif method == 'kernel KMeans':
        y_pred = kernel_method.Kernel_Kmeans(X,2)
    elif method == 'AP':
        y_pred = AP_clustering.AP_clustering(X)
    elif method == 'Spectral':
        y_pred = spectral_clustering.chose_para(X)

    return y_pred

def clustering_result(X, y, y_pred):
    print('同质性：%0.3f' % metrics.homogeneity_score(y, y_pred))  # 一个簇中只包含一个类别的样本，则满足同质性
    print('完整性：%0.3f' % metrics.completeness_score(y, y_pred))  # 同类别样本被归类到相同簇中，则满足完整性
    print('V-值： % 0.3f' % metrics.v_measure_score(y, y_pred))  # 同质性和完整性的加权平均
    print('调整后的兰德指数：%0.3f' % metrics.adjusted_rand_score(y, y_pred))  # [-1,1]调整后的兰德指数越大意味着聚类结果与真实情况越吻合
    print('调整后的互信息： %0.3f' % metrics.adjusted_mutual_info_score(y, y_pred))  # [0,1],越接近1表示差别越小

    # 如果没有标签的话，只能比较这一项
    print('轮廓系数：%0.3f' % metrics.silhouette_score(X, y_pred, metric='sqeuclidean'))  # [-1,1],越接近1表示差别越小

if __name__ == '__main__':
    X, y = load_data()
    y_pred = cluster(X,method='Spectral')
    clustering_result(X, y, y_pred)


