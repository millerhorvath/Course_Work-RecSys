import os
import cPickle as cpkl
import sys
import datetime
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import multiprocessing
import support_functions as sup


def kmeans_p(kmeans, matrix):
    return kmeans.fit(matrix)


num_cores = multiprocessing.cpu_count()

# file_pattern = sys.argv[2]
# file_ext = sys.argv[3]
file_pattern = 'MyTrain'
file_ext = 'csv'

data_path = sys.argv[1]
has_reader = 1
train_fold_path = os.path.join(data_path, 'train')

# Reading dataset size file
NUM_USERS, NUM_ITEMS = cpkl.load(open(os.path.join(data_path, 'dataset_size.pkl'), 'rb'))

# num_folds = int(sys.argv[6])
# num_clusters = int(sys.argv[7])
# cluster_list = [5, 10, 50, 100, 250, 500, 1000]
# cluster_list = [5, 10, 50, 100, 250, 500]
cluster_list = cpkl.load(open(os.path.join(data_path, 'cluster_list.pkl'), 'rb'))

log_file = sup.create_logfile(sys.argv[0])

file_base = file_pattern + '{}.' + file_ext

fold = 0

file = os.path.join(train_fold_path, file_base.format(fold))

# rat_dict = sup.read_ratings_csv_to_dict(file,True)

sup.logmsg('Creating ratings matrix for user-based and item-based clustering', log_file)

rating_matrix_user = sup.read_ratings_csv_to_matrix(file, True, NUM_USERS, NUM_ITEMS,
                                                    dtype=np.float)

rating_matrix_item = np.matrix(rating_matrix_user).getT()

sup.logmsg('Creating kmeans base structure', log_file)

kmeans = Parallel(n_jobs=num_cores, backend="threading") \
    (delayed(KMeans)(n_clusters=num_clusters, n_init=10, algorithm="full",
                     precompute_distances=True, n_jobs=1)
     for num_clusters in cluster_list)

sup.logmsg('Running user-based k-means', log_file)

clustered_data = Parallel(n_jobs=num_cores, backend="threading") \
    (delayed(kmeans_p)(km, rating_matrix_user) for km in kmeans)

sup.logmsg('Writing user-based kmeans data structure on disk', log_file)

for i in range(len(cluster_list)):
    num_clusters = cluster_list[i]
    write_file_name = 'user_{}_{}.cpkl'.format(num_clusters, fold)
    write_path = os.path.join(data_path, 'kmeans')

    sup.write_cpkl2(clustered_data[i], write_path, write_file_name)

sup.logmsg('Done with user-based', log_file)

sup.logmsg('Running item-based k-means', log_file)

clustered_data = Parallel(n_jobs=num_cores, backend="threading") \
    (delayed(kmeans_p)(km, rating_matrix_item) for km in kmeans)

sup.logmsg('Writing item-based kmeans data structure on disk', log_file)

for i in range(len(cluster_list)):
    num_clusters = cluster_list[i]
    write_file_name = 'item_{}_{}.cpkl'.format(num_clusters, fold)
    write_path = os.path.join(data_path, 'kmeans')

    sup.write_cpkl2(clustered_data[i], write_path, write_file_name)

sup.logmsg('Done with item-based', log_file)

sup.logmsg('Done with fold kmeans', log_file)
