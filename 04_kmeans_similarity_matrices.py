import os
import sys
import numpy as np
from joblib import Parallel, delayed
import cPickle as cpkl
import support_functions as sup


def p_loop(k, based, matrix, alg, metric, fold, output_folder, num_clusters,
           log_file):
    sup.logmsg('{}_{}_{}_k{}_c{}_f{} - Computing knn'.format(based, alg, metric,
                                                             k, num_clusters, fold), log_file)

    sim = []

    for i in range(num_clusters):
        if i in matrix.keys():
            n_samples = len(matrix[i])
            if k + 1 <= n_samples:
                sim.append(sup.similarity_sklearn(k, matrix[i], algorithm=alg,
                                                  metric=metric))
            else:
                sim.append(sup.similarity_sklearn(n_samples - 1, matrix[i], algorithm=alg,
                                                  metric=metric))
        else:
            sim.append(None)

    sup.logmsg('{}_{}_{}_k{}_c{}_f{} - Writing similarity matrix on disk'.format(
        based, alg, metric, k, num_clusters, fold), log_file)

    file_name = '{}_{}_{}_c{}_k{}_f{}.cpkl'.format(based, alg, metric, num_clusters,
                                                   k, fold)
    file_path = os.path.join(output_folder, 'kmeans_knn_sim_dicts')
    sup.write_cpkl2(sim, file_path, file_name)


data_path = sys.argv[1]
train_fold_path = os.path.join(data_path, 'train')

NUM_USERS, NUM_ITEMS = cpkl.load(open(os.path.join(data_path, 'dataset_size.pkl'), 'rb'))


# file_pattern = sys.argv[2]
# file_ext = sys.argv[3]
file_pattern = 'MyTrain'
file_ext = 'csv'

has_reader = 1
output_folder = data_path
num_folds = int(sys.argv[2])

cluster_list_folder = os.path.join(data_path, 'cluster_list')
# num_clusters = int(sys.argv[7])
# cluster_list = [5, 10, 50, 100, 250, 500, 1000]
cluster_list = cpkl.load(open(os.path.join(data_path, 'cluster_list.pkl'), 'rb'))
k_list = cpkl.load(open(os.path.join(data_path, 'k_list.pkl'), 'rb'))
alg_list = ['brute']
metric_list = ['cosine']

log_file = sup.create_logfile(sys.argv[0])

file_base = file_pattern + '{}.' + file_ext

userid_to_idx = {}
itemid_to_idx = {}
idx_to_userid = {}
idx_to_itemid = {}

for fold in range(num_folds):
    file = os.path.join(train_fold_path, file_base.format(fold))

    rating_matrix_user = sup.read_ratings_csv_to_matrix(file, True, NUM_USERS,
                                                        NUM_ITEMS, dtype=np.float)
    # rating_matrix_item = np.matrix(rating_matrix_user).getT()
    rating_matrix_item = sup.transpose_matrix(rating_matrix_user)

    c_list = {}
    cluster_file_base = '{}_{}_{}.cpkl'

    for based in ['user', 'item']:
        for num_clusters in cluster_list:
            file_name = cluster_file_base.format(based, num_clusters, fold)
            file = os.path.join(cluster_list_folder, file_name)
            c_list[(based, num_clusters)] = sup.read_cpkl(file)

    p_list2 = []

    for based in [('user', rating_matrix_user), ('item', rating_matrix_item)]:
        for num_clusters in cluster_list:
            p_list2.append((based, num_clusters))

    split_matrices = Parallel(n_jobs=-1, backend="threading") \
        (delayed(sup.split_matrix_by_cluster)(c_list[(p[0][0], p[1])], p[0][1])
         for p in p_list2)

    split_mat_dict = {}

    for i in range(len(p_list2)):
        p = p_list2[i]
        key = (p[0][0], p[1])

        split_mat_dict[key] = split_matrices[i][0]

    p_list = []

    for based in ['user', 'item']:
        for alg in alg_list:
            for metric in metric_list:
                for k in k_list:
                    for num_clusters in cluster_list:
                        p_list.append([k, based, alg, metric, num_clusters])

    Parallel(n_jobs=-1, backend="threading")(delayed(p_loop)(p[0], p[1],
                                                             split_mat_dict[(p[1], p[4])],
                                                             p[2], p[3], fold,
                                                             output_folder, p[4],
                                                             log_file)
                                             for p in p_list)
