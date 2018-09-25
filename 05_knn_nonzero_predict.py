import os
import cPickle as cpkl
import sys
import datetime
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import support_functions as sup
import math


def p_loop(k, based, rat_matrix, alg, metric, fold, output_folder, test_dict,
           sim_mat_path, user_avg, item_avg):
    pred_dict = {}
    coverage = float(len(test_dict))

    file_name = '{}_{}_{}_k{}_f{}.cpkl'.format(based, alg, metric, k, fold)

    sup.logmsg('{}_{}_{}_k{}_f{} - Predicting ratings'.format(based, alg, metric, k,
                                                              fold), log_file)

    file = os.path.join(sim_mat_path, file_name)
    sim_dict = sup.read_cpkl(file)

    if based == 'user':
        for key in sorted(test_dict.keys()):
            userId, itemId = key
            rat = test_dict[key]

            pred_rat = sup.ub_predict_par(userId, itemId, sim_dict, rat_matrix)

            if pred_rat == 0.:
                coverage -= 1.
                if not math.isnan(item_avg[itemId]):
                    pred_rat = item_avg[itemId]
                else:
                    # sup.logmsg('COV_ERROR - User average used on user-based model.',
                    # 	log_file)
                    pred_rat = user_avg[userId]

            pred_dict[key] = (rat, pred_rat)
    elif based == 'item':
        for key in sorted(test_dict.keys()):
            userId, itemId = key
            rat = test_dict[key]

            pred_rat = sup.ib_predict_par(userId, itemId, sim_dict, rat_matrix)

            if pred_rat == 0.:
                coverage -= 1.
                pred_rat = user_avg[userId]

            pred_dict[key] = (rat, pred_rat)

    coverage /= float(len(test_dict))
    pred_dict[-1] = coverage

    sup.logmsg('{}_{}_{}_k{}_f{} - Writing ratings on disk'.format(based, alg,
                                                                   metric, k, fold), log_file)

    file_path = os.path.join(output_folder, 'knn_predictions')
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
sim_mat_path = os.path.join(data_path, 'knn_sim_dicts')
output_folder = data_path
num_folds = int(sys.argv[2])


NUM_USERS, NUM_ITEMS = cpkl.load(open(os.path.join(data_path, 'dataset_size.pkl'), 'rb'))
k_list = cpkl.load(open(os.path.join(data_path, 'k_list.pkl'), 'rb'))

alg_list = ['brute', 'ball_tree', 'kd_tree']
metric_list = ['cosine', 'euclidean', 'minkowski']

log_file = sup.create_logfile(sys.argv[0])

file_base = file_pattern + '{}.' + file_ext
file_base2 = file_pattern2 + '{}.' + file_ext

for fold in range(num_folds):
    file = os.path.join(train_fold_path, file_base.format(fold))

    rat_matrix = sup.read_ratings_csv_to_matrix(file, True, NUM_USERS, NUM_ITEMS,
                                                dtype=np.float)

    file = os.path.join(test_fold_path, file_base2.format(fold))
    test_dict = sup.read_ratings_csv_to_dict(file, True)

    user_avg = np.true_divide(rat_matrix.sum(1), (rat_matrix != 0).sum(1))
    item_avg = np.true_divide(rat_matrix.sum(0), (rat_matrix != 0).sum(0))

    p_list = []

    for based in ['user', 'item']:
        for alg in alg_list:
            for metric in metric_list:
                for k in k_list:
                    if metric != 'cosine' or alg == 'brute':
                        p_list.append([k, based, alg, metric])

    Parallel(n_jobs=-1, backend="threading")(delayed(p_loop)(p[0], p[1],
                                                             rat_matrix, p[2], p[3], fold, output_folder, test_dict,
                                                             sim_mat_path,
                                                             user_avg, item_avg)
                                             for p in p_list)
