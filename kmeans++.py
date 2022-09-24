import numpy as np


def init_centroid(X, n_data, k):
    # 1つ目のセントロイドをランダムに選択
    idx = np.random.choice(n_data, 1)
    centroids = X[idx]
    for i in range(k - 1):
        # 各データ点とセントロイドとの距離を計算
        distances = compute_distances(X, len(centroids), n_data, centroids)
        # 各データ点と最も小さいセントロイドとの距離の二乗を計算
        closest_dist_sq = np.min(distances ** 2, axis=1)
        # 距離の二乗の和を計算
        weights = closest_dist_sq.sum()
        # [0, 1)の乱数と距離の二乗和を掛ける
        rand_vals = np.random.random_sample() * weights
        # 距離の二乗の累積和を計算し、rand_valと最も値が近いデータ点のindexを取得
        candidate_idx = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
        # 選ばれた点を新たなセントロイドとして追加
        centroids = np.vstack([centroids, X[candidate_idx]])
        return centroids


def compute_distances(X, k, n_data, centroids):
    distances = np.zeros((n_data, k))
    for idx_centroids in range(k):
        dist = np.sqrt(np.sum((X - centroids[idx_centroids]) ** 2, axis=1))
        distances[:, idx_centroids] = dist
    return distances


def k_means(X, k, max_iter=300):
    """
    X.shape = (データ数, 次元数)
    k = クラスタ数
    """
    n_data, n_features = X.shape
    # セントロイドの初期値
    centroids = init_centroid(X, k, n_data)
    # 新しいクラスタを格納するための配列
    new_cluster = np.zeros(n_data)
    # 各データの所属クラスタを保存する配列
    cluster = np.zeros(n_data)

    for epoch in range(max_iter):
        # 各データ点のセントロイドとの距離を計算
        distances = compute_distances(X, k, n_data, centroids)
        # 新たな所属クラスタを計算
        new_cluster = np.argmin(distances, axis=1)
        # 全てのクラスタに対してセントロイドを再計算
        for idx_centroids in range(k):
            centroids[idx_centroids] = X[new_cluster == idx_centroids].mean(axis=0)
        # クラスタによるグループ分けに変化がなかったら終了
        if (new_cluster == cluster).all():
            break
        cluster = new_cluster
    return cluster


# probabilities = np.repeat(1/n, n)
# centroids = np.zeros((k, 2))
# distances = np.zeros((n, k))
#
# for i in range(k):
#     # probabilitiesの確率に従ってセントロイドとなる点をdataから選ぶ
#     centroids[i] = data[np.random.choice(np.arange(n), p=probabilities, size=(1))]
#     # dataとcentroidsの距離の二乗を取る
#     distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)
#     # probabilitiesを0~1の値に正規化する
#     probabilities =
