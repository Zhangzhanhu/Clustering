# -*- encoding: utf-8 -*-
"""
@Modify Time    2021/5/26 9:09
@Author         Tunan
@Desciption
                谱聚类
可以直接调用函数SpectralClustering
因此在做一步参数选择之后直接聚类即可
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import calinski_harabasz_score


def chose_para(X):
    # 默认使用的是高斯核，需要对n_cluster和gamma进行调参，选择合适的参数
    scores = []
    s = dict()
    for index, gamma in enumerate((0.01, 0.1, 1, 10)):
        for index, k in enumerate((2, 3, 4)):
            y_pred = SpectralClustering(n_clusters=k,gamma=gamma).fit_predict(X.data)
            print("Calinski-Harabasz Score with gamma=", gamma, "n_cluster=", k, "score=",
                  calinski_harabasz_score(X.data, y_pred))
            tmp = dict()
            tmp['gamma'] = gamma
            tmp['n_cluster'] = k
            tmp['score'] = calinski_harabasz_score(X.data, y_pred)
            s[calinski_harabasz_score(X.data, y_pred)] = tmp
            scores.append(calinski_harabasz_score(X.data, y_pred))

    max_score = s.get(np.max(scores))
    print("max score:\n",max_score)

    gamma = list(max_score.values())[0]
    n_clusters = list(max_score.values())[1]

    y_pred = SpectralClustering(n_clusters=n_clusters,gamma=gamma).fit_predict(X)
    plt.title('SpectralClustering of blobs')
    plt.scatter(X[:, 0], X[:, 1], marker='.',c=y_pred)
    plt.show()

    return y_pred





