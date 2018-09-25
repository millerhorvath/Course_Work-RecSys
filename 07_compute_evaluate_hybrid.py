import os
import sys
from joblib import Parallel, delayed
import support_functions as sup
import random
import cPickle as cpkl


def p_loop(alg, metric, k, user_cluster, item_cluster, alpha, eval, l, log_file):
    key = (alg, metric, k, user_cluster, item_cluster)

    old_a = [alpha[key][0], alpha[key][1], alpha[key][2], alpha[key][3]]
    size = len(alpha[key])
    g = random.randint(0, size - 1)

    t = random.randint(0, size - 1)
    while g == t:
        t = random.randint(0, size - 1)

    w = random.random() * alpha[key][g] * l

    alpha[key][g] = alpha[key][g] - w
    alpha[key][t] = alpha[key][t] + w

    new_eval = sup.eval_for_hybrid_singular(pred, alpha, key)

    if new_eval[0] < eval[key, 0][0]:
        eval[key, 0] = new_eval
        sup.logmsg('{} - {} NEW BEST TRADITIONAL ALPHA\n{}\n'.format(key, alpha[key],
                                                                     new_eval),
                   log_file)
    else:
        alpha[key] = [old_a[0], old_a[1], old_a[2], old_a[3]]

    for uc in range(user_cluster):
        for ic in range(item_cluster):
            key2 = (alg, metric, k, user_cluster, item_cluster, uc, ic)

            old_a = [alpha[key2][0], alpha[key2][1], alpha[key2][2], alpha[key2][3]]
            size = len(alpha[key2])
            g = random.randint(0, size - 1)

            t = random.randint(0, size - 1)
            while g == t:
                t = random.randint(0, size - 1)

            w = random.random() * alpha[key2][g] * l

            alpha[key2][g] = alpha[key2][g] - w
            alpha[key2][t] = alpha[key2][t] + w

            new_eval = sup.eval_for_hybrid_mutable(pred, alpha, key, c_list)

            if new_eval[0] < eval[key, 1][0]:
                eval[key, 1] = new_eval
                sup.logmsg('{} - {} NEW BEST CLUSTER-SPECIALIZED ALPHA\n{}\n'.format(key2,
                                                                                     alpha[key2],
                                                                                     new_eval),
                           log_file)
            else:
                alpha[key2] = [old_a[0], old_a[1], old_a[2], old_a[3]]


data_path = sys.argv[1]
test_fold_path = os.path.join(data_path, 'test')
knn_pred_path = os.path.join(data_path, 'knn_predictions')
kmeans_pred_path = os.path.join(data_path, 'kmeans_predictions')
cluster_list_folder = os.path.join(data_path, 'cluster_list')
output_folder = data_path

NUM_USERS, NUM_ITEMS = cpkl.load(open(os.path.join(data_path, 'dataset_size.pkl'), 'rb'))

# Default hybrid model parameters
k_list = cpkl.load(open(os.path.join(data_path, 'k_list.pkl'), 'rb'))
cluster_list_2 = cpkl.load(open(os.path.join(data_path, 'cluster_list.pkl'), 'rb'))

cluster_list = {}
cluster_list['user'] = cluster_list_2
cluster_list['item'] = cluster_list_2

# # # # # MODIFY THE FOLLOWING 3 COMMENTED LINES TO CUSTOMIZE THE HYBRID MODEL PARAMETERS
# cluster_list['user'] = [5]
# cluster_list['item'] = [5]
# k_list = [50]

# # The following 2 lines can be modified as well in order to customized the hybrid model
alg_list = ['brute']
metric_list = ['cosine']  # Best similarity measure in the standalone models evaluation

log_file = sup.create_logfile(sys.argv[0])

# test_file = "myTest0.csv"
file_base_knn = os.path.join(knn_pred_path, "{}_{}_{}_k{}_f0.cpkl")
file_base_kmeans = os.path.join(kmeans_pred_path, "{}_{}_{}_c{}_k{}_f0.cpkl")
file_base_cluster_list = os.path.join(cluster_list_folder, "{}_{}_0.cpkl")

alpha = {}
pred = {}
h_pred = {}
c_list = {}
eval = {}

p_list = []

for alg in alg_list:
    for metric in metric_list:
        if metric != 'cosine' or alg == 'brute':
            for k in k_list:
                p_list.append((alg, metric, k))

for p in p_list:
    for user_cluster in cluster_list['user']:
        for item_cluster in cluster_list['item']:
            key = (p[0], p[1], p[2], user_cluster, item_cluster)
            alpha[key] = [0.25, 0.25, 0.25, 0.25]  # Traditional alpha coefficients

            # Cluster-Specialized alpha coefficients
            for uc in range(user_cluster):
                for ic in range(item_cluster):
                    key = (p[0], p[1], p[2], user_cluster, item_cluster, uc, ic)
                    alpha[key] = [0.25, 0.25, 0.25, 0.25]

p_list1 = []

for based in ['user', 'item']:
    for p in p_list:
        p_list1.append((based, p[0], p[1], p[2]))

sup.logmsg('Reading kNN predictions', log_file)

file = Parallel(n_jobs=-1, backend="threading") \
    (delayed(sup.read_cpkl)(file_base_knn.format(p[0], p[1], p[2], p[3]))
     for p in p_list1)

sup.logmsg('Done reading kNN predictions', log_file)

for p in p_list1:
    pred[p] = file.pop(0)

    # Removing coverage from the dictionary
    pred[p].pop(-1)

p_list2 = []

for p in p_list1:
    for num_cluster in cluster_list[p[0]]:
        p_list2.append((p[0], p[1], p[2], p[3], num_cluster))

sup.logmsg('Reading k-means predictions', log_file)

file = Parallel(n_jobs=-1, backend="threading") \
    (delayed(sup.read_cpkl)(file_base_kmeans.format(p[0], p[1], p[2], p[4], p[3]))
     for p in p_list2)

sup.logmsg('Done reading k-means predictions', log_file)

for p in p_list2:
    pred[p] = file.pop(0)

    # Removing coverage from the dictionary
    pred[p].pop(-1)

for based in ['user', 'item']:
    for num_cluster in cluster_list[based]:
        c_list[based, num_cluster] = sup.read_cpkl(file_base_cluster_list.format(based,
                                                                                 num_cluster))

for p in p_list:
    for user_cluster in cluster_list['user']:
        for item_cluster in cluster_list['item']:
            key = (p[0], p[1], p[2], user_cluster, item_cluster)
            eval[key, 0] = sup.eval_for_hybrid_singular(pred, alpha, key)

            for uc in range(user_cluster):
                for ic in range(item_cluster):
                    eval[key, 1] = sup.eval_for_hybrid_mutable(pred, alpha, key, c_list)

l = 0.1

p_list3 = []

for p in p_list:
    for user_cluster in cluster_list['user']:
        for item_cluster in cluster_list['item']:
            p_list3.append((p[0], p[1], p[2], user_cluster, item_cluster))

for i in range(1000):
    if (i + 1) % 10 == 0:
        sup.logmsg('{}% of iterations done'.format(float(i) / 10.0), log_file)

    Parallel(n_jobs=-1, backend="threading") \
        (delayed(p_loop)(p[0], p[1], p[2], p[3], p[4], alpha, eval, l, log_file)
         for p in p_list3)

file_path = os.path.join(output_folder, 'hybrid_evaluation')
file = 'hybrid_eval.cpkl'
sup.write_cpkl2(eval, file_path, file)
file = 'hybrid_alpha.cpkl'
sup.write_cpkl2(alpha, file_path, file)

file = os.path.join(output_folder, 'hybrid_eval.csv')
f = open(file, 'w')

f.write('alg; metric; k; user_clusters; item_clusters; hybrid;;rmse; mse; mae\n')

for p in p_list:
    for user_cluster in cluster_list['user']:
        for item_cluster in cluster_list['item']:
            key = (p[0], p[1], p[2], user_cluster, item_cluster)
            alg, metric, k = p[0], p[1], p[2]
            rmse, mse, mae = eval[key, 0]
            f.write('{};{};{};{};{};traditional;;{};{};{}\n'.format(alg, metric, k, user_cluster,
                                                                    item_cluster, rmse, mse, mae))
            rmse, mse, mae = eval[key, 1]
            f.write('{};{};{};{};{};cluster-specialized;;{};{};{}\n'.format(alg, metric, k, user_cluster,
                                                                            item_cluster, rmse, mse, mae))

f.close()
