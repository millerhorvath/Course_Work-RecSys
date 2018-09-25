import os
import cPickle as cpkl
import sys
import datetime
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import support_functions as sup

def ratings_by_cluster(kmeans, matrix, based, k, fold, log_file):
	sup.logmsg('{}_k{}_f{} - Predicting clusters'.format(based, k, fold), log_file)
	
	cluster_list = kmeans.predict(matrix)

	sup.logmsg('{}_k{}_f{} - Done predicting clusters'.format(based, k, fold),
		log_file)

	return cluster_list


data_path = sys.argv[1]
train_fold_path = os.path.join(data_path, 'train')

# Reading dataset size file

NUM_USERS, NUM_ITEMS = cpkl.load(open(os.path.join(data_path, 'dataset_size.pkl'), 'rb'))

# file_pattern = sys.argv[2]
# file_ext = sys.argv[3]
file_pattern = 'MyTrain'
file_ext = 'csv'

has_reader = 1
num_folds = int(sys.argv[2])
cluster_files_path = sys.argv[3]


# num_clusters = int(sys.argv[8])
# cluster_list = [5, 10, 50, 100, 250, 500, 1000]
cluster_list = cpkl.load(open(os.path.join(data_path, 'cluster_list.pkl'), 'rb'))

log_file = sup.create_logfile(sys.argv[0])

file_base = file_pattern + '{}.' + file_ext

p_list = []

for based in ['user', 'item']:
		for num_clusters in cluster_list:
			p_list.append( (based, num_clusters) )

sup.logmsg('Reading kmeans files from disk', log_file)

kmeans = {}

for p in p_list:
	based = p[0]
	num_clusters = p[1]

	file = os.path.join(cluster_files_path, '{}_{}_0.cpkl'.format(based,
		num_clusters))
	kmeans[(based, num_clusters)] = sup.read_cpkl(file)

for fold in range(num_folds):
	sup.logmsg('Reading training fold {}'.format(fold), log_file)
	
	file = os.path.join(train_fold_path, file_base.format(fold))

	rating_matrix = {}
	rating_matrix['user'] = sup.read_ratings_csv_to_matrix(file, True, NUM_USERS,
		NUM_ITEMS, dtype=np.float)
	rating_matrix['item'] = np.matrix(rating_matrix['user']).getT()
	
	rat_by_cluster = Parallel(n_jobs=-1, backend="threading")\
		(delayed(ratings_by_cluster)(kmeans=kmeans[(p[0], p[1])],
			matrix=rating_matrix[(p[0])], based=p[0], k=p[1], fold=fold,
			log_file=log_file)
		for p in p_list)

	sup.logmsg('Writing clustered ratings on disk', log_file)

	# out_file_base = '{}_{}.csv'
	out_file_base = '{}_{}_{}.cpkl'
	
	# for num_clusters in cluster_list:
	for i in range(len(p_list)):
		based = p_list[i][0]
		num_clusters = p_list[i][1]
		c_list = rat_by_cluster[i]
		
		write_path = os.path.join(data_path, 'cluster_list')
		file = out_file_base.format(based, num_clusters, fold)

		sup.write_cpkl2(c_list, write_path, file)

