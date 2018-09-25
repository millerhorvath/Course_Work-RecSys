import os
import sys
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import support_functions as sup
import math
import cPickle as cpkl


def p_loop(k, based, rat_matrix, alg, metric, fold, output_folder, test_dict,
           sim_mat_path, user_avg, item_avg, log_file, id_to_idx, num_clusters):
    pred_dict = {}
    coverage = float(len(test_dict))

    file_name = '{}_{}_{}_c{}_k{}_f{}.cpkl'.format(based, alg, metric, num_clusters,
                                                   k, fold)

    sup.logmsg('{}_{}_{}_c{}_k{}_f{} - Predicting ratings'.format(based, alg, metric,
                                                                  num_clusters, k, fold),
                                                                  log_file)

    file = os.path.join(sim_mat_path, file_name)

    sim_dict = sup.read_cpkl(file)

    if based == 'user':
        for key in sorted(test_dict.keys()):
            userId, itemId = key
            rat = test_dict[key]

            c, userIdx = id_to_idx[userId]

            if sim_dict[c] != None:
                pred_rat = sup.ub_predict_par(userIdx, itemId, sim_dict[c],
                                              rat_matrix[c])
            else:
                pred_rat = 0.

            if pred_rat == 0.:
                coverage -= 1.
                if not math.isnan(item_avg[itemId]):
                    pred_rat = item_avg[itemId]
                else:
                    pred_rat = user_avg[userId]

            pred_dict[key] = (rat, pred_rat)

    elif based == 'item':
        for key in sorted(test_dict.keys()):
            userId, itemId = key
            rat = test_dict[key]

            c, itemIdx = id_to_idx[itemId]

            if sim_dict[c] != None:
                pred_rat = sup.ib_predict_par(userId, itemIdx, sim_dict[c],
                                              rat_matrix[c])
            else:
                pred_rat = 0.

            if pred_rat == 0.:
                coverage -= 1.
                pred_rat = user_avg[userId]

            pred_dict[key] = (rat, pred_rat)

    coverage /= float(len(test_dict))
    pred_dict[-1] = coverage

    sup.logmsg('{}_{}_{}_c{}_k{}_f{} - Writing ratings on disk'.format(based, alg,metric,
                                                                       num_clusters, k,
                                                                       fold), log_file)

    file_path = os.path.join(output_folder, 'kmeans_predictions')
    sup.write_cpkl2(pred_dict, file_path, file_name)
    pred_dict.clear()


num_cores = multiprocessing.cpu_count()

data_path = sys.argv[1]

train_fold_path = os.path.join(data_path, 'train')
file_pattern = 'MyTrain'
file_ext = 'csv'

test_fold_path = os.path.join(data_path, 'test')
file_pattern2 = 'MyTest'

has_reader = 1
sim_mat_path = os.path.join(data_path, 'kmeans_knn_sim_dicts')
output_folder = data_path
num_folds = int(sys.argv[2])
cluster_list_folder = os.path.join(data_path, 'cluster_list')

NUM_USERS, NUM_ITEMS = cpkl.load(open(os.path.join(data_path, 'dataset_size.pkl'), 'rb'))
k_list = cpkl.load(open(os.path.join(data_path, 'k_list.pkl'), 'rb'))
cluster_list = cpkl.load(open(os.path.join(data_path, 'cluster_list.pkl'), 'rb'))

alg_list = ['brute']
metric_list = ['cosine']

log_file = sup.create_logfile(sys.argv[0])

file_base = file_pattern + '{}.' + file_ext
file_base2 = file_pattern2 + '{}.' + file_ext

rat_matrix = {}

for fold in range(num_folds):
    file = os.path.join(train_fold_path, file_base.format(fold))

    rat_matrix['user'] = sup.read_ratings_csv_to_matrix(file, True, NUM_USERS,
                                                        NUM_ITEMS, dtype=np.float)
    rat_matrix['item'] = sup.transpose_matrix(rat_matrix['user'])

    file = os.path.join(test_fold_path, file_base2.format(fold))
    test_dict = sup.read_ratings_csv_to_dict(file, True)

    user_avg = np.true_divide(rat_matrix['user'].sum(1),
                              (rat_matrix['user'] != 0).sum(1))
    item_avg = np.true_divide(rat_matrix['user'].sum(0),
                              (rat_matrix['user'] != 0).sum(0))

    file_base_c_list = '{}_{}_{}.cpkl'

    id_to_idx = {}

    for based in ['user', 'item']:
        for num_clusters in cluster_list:
            file_name = file_base_c_list.format(based, num_clusters, fold)
            file = os.path.join(cluster_list_folder, file_name)
            c_list = sup.read_cpkl(file)

            key = (based, num_clusters)

            (rat_matrix[key], id_to_idx[key],
             idx_to_id) = sup.split_matrix_by_cluster(c_list, rat_matrix[based])

            if based == 'item':
                for c in range(num_clusters):
                    if c in rat_matrix[key].keys():
                        rat_matrix[key][c] = sup.transpose_matrix(rat_matrix[key][c])

    p_list = []

    for alg in alg_list:
        for metric in metric_list:
            if metric != 'cosine' or alg == 'brute':
                for based in ['user', 'item']:
                    for k in k_list:
                        for num_clusters in cluster_list:
                            p_list.append([k, based, alg, metric, num_clusters])

    Parallel(n_jobs=-1, backend="threading")(delayed(p_loop)(p[0], p[1],
                                                             rat_matrix[(p[1], p[4])],
                                                             p[2], p[3], fold,
                                                             output_folder, test_dict,
                                                             sim_mat_path, user_avg, item_avg,
                                                             log_file,
                                                             id_to_idx[(p[1], p[4])],
                                                             p[4])
                                             for p in p_list)
