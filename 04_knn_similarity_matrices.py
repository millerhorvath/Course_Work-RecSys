import os
import cPickle as cpkl
import sys
import datetime
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import support_functions as sup


def p_loop(k, based, matrix, alg, metric, fold, output_folder):
    sup.logmsg('{}_{}_{}_k{}_f{} - Computing knn'.format(based, alg, metric, k,
                                                         fold))
    user_sim = sup.similarity_sklearn(k, matrix, algorithm=alg, metric=metric)

    sup.logmsg('{}_{}_{}_k{}_f{} - Writing similarity matrix on disk'.format(based, alg, metric, k,
                                                         fold))

    file_name = '{}_{}_{}_k{}_f{}.cpkl'.format(based, alg, metric, k, fold)
    file_path = os.path.join(output_folder, 'knn_sim_dicts')
    sup.write_cpkl2(user_sim, file_path, file_name)
    user_sim.clear()


num_cores = multiprocessing.cpu_count()

# file_pattern = sys.argv[2]
# file_ext = sys.argv[3]
file_pattern = 'MyTrain'
file_ext = 'csv'

data_path = sys.argv[1]
train_fold_path = os.path.join(data_path, 'train')
has_reader = 1
output_folder = data_path
num_folds = int(sys.argv[2])

NUM_USERS, NUM_ITEMS = cpkl.load(open(os.path.join(data_path, 'dataset_size.pkl'), 'rb'))
k_list = cpkl.load(open(os.path.join(data_path, 'k_list.pkl'), 'rb'))
alg_list = ['brute', 'ball_tree', 'kd_tree']
metric_list = ['cosine', 'euclidean', 'minkowski']

log_file = sup.create_logfile(sys.argv[0])

file_base = file_pattern + '{}.' + file_ext

p_list = []

for based in [('user'), ('item')]:
    for alg in alg_list:
        for metric in metric_list:
            for k in k_list:
                if metric != 'cosine' or alg == 'brute':
                    p_list.append([k, based, alg, metric])

rating_matrix = {}

for fold in range(num_folds):
    file = os.path.join(train_fold_path, file_base.format(fold))

    rating_matrix['user'] = sup.read_ratings_csv_to_matrix(file, True, NUM_USERS,
                                                           NUM_ITEMS)
    rating_matrix['item'] = sup.transpose_matrix(rating_matrix['user'])

    Parallel(n_jobs=-1, backend="threading")(delayed(p_loop)(p[0], p[1],
                                                             rating_matrix[p[1]], p[2], p[3], fold, output_folder)
                                             for p in p_list)
