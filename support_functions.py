import os
import numpy as np
import cPickle as cpkl
import datetime
from sklearn.neighbors import NearestNeighbors
from math import sqrt


def eval_for_hybrid_mutable(pred, alpha, key, c_list):
    alg, metric, k, user_clusters, item_clusters = key
    ubknn_pred = pred[('user', alg, metric, k)]
    ibknn_pred = pred[('item', alg, metric, k)]
    ubkm_pred = pred[('user', alg, metric, k, user_clusters)]
    ibkm_pred = pred[('item', alg, metric, k, item_clusters)]

    user_cluster_list = c_list['user', user_clusters]
    item_cluster_list = c_list['item', item_clusters]

    alpha_ubknn = {}
    alpha_ibknn = {}
    alpha_ubkm = {}
    alpha_ibkm = {}

    for uc in range(user_clusters):
        for ic in range(item_clusters):
            key2 = (alg, metric, k, user_clusters, item_clusters, uc, ic)
            alpha_ubknn[uc, ic] = alpha[key2][0]
            alpha_ibknn[uc, ic] = alpha[key2][1]
            alpha_ubkm[uc, ic] = alpha[key2][2]
            alpha_ibkm[uc, ic] = alpha[key2][3]

    mse = 0.0
    mae = 0.0
    size = 0.0

    for k in ubknn_pred.keys():
        userid, itemid = k

        uc = user_cluster_list[userid]
        ic = item_cluster_list[itemid]

        (rat, ubknn) = ubknn_pred[k]
        (rat, ibknn) = ibknn_pred[k]
        (rat, ubkm) = ubkm_pred[k]
        (rat, ibkm) = ibkm_pred[k]

        pred_rat = alpha_ubknn[uc, ic] * ubknn + alpha_ibknn[uc, ic] * ibknn +\
                   alpha_ubkm[uc, ic] * ubkm + alpha_ibkm[uc, ic] * ibkm

        err = pred_rat - float(rat)
        mse += (err * err)
        mae += abs(err)
        size += 1.0

    mse /= size
    mae /= size
    rmse = sqrt(mse)

    return rmse, mse, mae


def eval_for_hybrid_singular(pred, alpha, key):
    alg, metric, k, user_clusters, item_clusters = key
    ubknn_pred = pred[('user', alg, metric, k)]
    ibknn_pred = pred[('item', alg, metric, k)]
    ubkm_pred = pred[('user', alg, metric, k, user_clusters)]
    ibkm_pred = pred[('item', alg, metric, k, item_clusters)]

    alpha_ubknn = alpha[key][0]
    alpha_ibknn = alpha[key][1]
    alpha_ubkm = alpha[key][2]
    alpha_ibkm = alpha[key][3]

    mse = 0.0
    mae = 0.0
    size = 0.0

    for k in ubknn_pred.keys():
        (rat, ubknn) = ubknn_pred[k]
        (rat, ibknn) = ibknn_pred[k]
        (rat, ubkm) = ubkm_pred[k]
        (rat, ibkm) = ibkm_pred[k]

        pred_rat = alpha_ubknn * ubknn + alpha_ibknn * ibknn +\
                   alpha_ubkm * ubkm + alpha_ibkm * ibkm

        err = pred_rat - float(rat)
        mse += (err * err)
        mae += abs(err)
        size += 1.0

    mse /= size
    mae /= size
    rmse = sqrt(mse)

    return rmse, mse, mae


def split_matrix_by_cluster(c_list, rating_matrix):
    clustered_rating_matrix = {}

    id_to_idx = {}
    idx_to_id = {}

    for i in range(len(rating_matrix)):
        c = c_list[i]

        if c not in clustered_rating_matrix.keys():
            clustered_rating_matrix[c] = []

        idx = len(clustered_rating_matrix[c])
        id_to_idx[i] = (c, idx)
        idx_to_id[c, idx] = i

        clustered_rating_matrix[c].append(rating_matrix[i])

    return (clustered_rating_matrix, id_to_idx, idx_to_id)


def ub_predict_par(userid, itemid, sim_dict, rat_matrix):
    sum_rating = 0.
    sum_similarity = 0.

    for i in range(1, len(sim_dict[userid])):
        nei, sim = sim_dict[userid][i]

        rat = rat_matrix[nei][itemid]

        if rat > 0:
            sim = 1. / (1. + sim)
            sum_rating += rat * sim
            sum_similarity += sim

    if sum_rating > 0:
        return (sum_rating / sum_similarity)
    else:
        return 0.


def ib_predict_par(userid, itemid, sim_dict, rat_matrix):
    sum_rating = 0.
    sum_similarity = 0.

    for i in range(1, len(sim_dict[itemid])):
        nei, sim = sim_dict[itemid][i]

        rat = rat_matrix[userid][nei]

        if rat > 0:
            sim = 1. / (1.+sim)
            sum_rating += rat * sim
            sum_similarity += sim

    if sum_rating > 0:
        return (sum_rating / sum_similarity)
    else:
        return 0.


def similarity_sklearn(top_n, matrix, algorithm='brute', metric='cosine'):
    # minkowski
    nbrs = NearestNeighbors(n_neighbors=top_n+1, algorithm=algorithm, metric=metric,
        n_jobs=1).fit(matrix)

    # indices for nearest neighbors and their distances
    distances, indices = nbrs.kneighbors(matrix)
    similars = {}

    for i in range(len(distances)):
        sim_list = []
        
        for nei in range(1, top_n+1):
            nei_id = indices[i][nei]
            sim = distances[i][nei]
            sim_list.append([nei_id, sim])
        similars[i] = sim_list[:]
    return similars


def clustered_similar_matrix(k, based, matrix, alg, metric, fold, num_clusters, log_file):
    logmsg('{}_{}_{}_k{}_c{}_f{} - Computing knn'.format(based, alg, metric,
                                                             k, num_clusters, fold), log_file)

    sim = {}

    for i in matrix.keys():
        n_samples = len(matrix[i])
        if k + 1 <= n_samples:
            sim[i] = similarity_sklearn(k, matrix[i], algorithm=alg, metric=metric)
        else:
            sim[i] = similarity_sklearn(n_samples - 1, matrix[i], algorithm=alg,
                                        metric=metric)

    return sim


def transpose_matrix(matrix):
    lines = len(matrix)
    columns = len(matrix[0])

    t_matrix = np.zeros((columns, lines))

    for i in range(lines):
        for j in range(columns):
            t_matrix[j][i] = matrix[i][j]

    return t_matrix


def read_cpkl(file):
    f = open(file, 'rb')
    data = cpkl.load(f)
    f.close()

    return data


def write_cpkl2(data, path, file):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass

    f = open(os.path.join(path, file), 'wb')
    cpkl.dump(data, f)
    f.close()


def write_cpkl(data, file):
    f = open(file, 'wb')
    cpkl.dump(data, f)
    f.close()


def read_ratings_csv_to_matrix(file, has_reader, num_users, num_items, sep=',',
                        dtype=np.float):
    f = open(file, 'r')

    if has_reader:
        f.readline()

    data = f.readlines()

    f.close()

    rat_matrix = np.zeros((num_users, num_items), dtype=dtype)

    for i in range(len(data)):
        data[i] = data[i].split(sep)
        userId = int(data[i][0])
        itemId = int(data[i][1])
        rating = dtype(data[i][2])

        rat_matrix[userId][itemId] = rating

    return rat_matrix


def read_ratings_csv_to_dict(file, has_reader, sep=',', dtype=np.float):
    f = open(file, 'r')

    if has_reader:
        f.readline()

    data = f.readlines()

    f.close()

    rat_dict = {}

    for i in range(len(data)):
        data[i] = data[i].split(sep)
        userId = int(data[i][0])
        itemId = int(data[i][1])
        rating = dtype(data[i][2])

        rat_dict[(userId, itemId)] = rating

    return rat_dict


def logmsg(args, file=None):
    msg = '[' + str(datetime.datetime.now()) + ']' + args
    print msg

    if file:
        file.write(msg + '\n')


def create_logfile(file, path="log"):
    time = str(datetime.datetime.now()).replace(':', '-')

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass

    file_path = file + '_[' + time + '].log'
    f = open(os.path.join(path, file_path), 'w')
    return f
