import os
import sys
from joblib import Parallel, delayed
import multiprocessing
import support_functions as sup
from math import sqrt
import cPickle as cpkl


def p_loop(based, alg, metric, k, fold, output_folder, pred_path, log_file):
    file_name = '{}_{}_{}_k{}_f{}.cpkl'.format(based, alg, metric, k, fold)

    sup.logmsg('{}_{}_{}_k{}_f{} - Evaluating ratings'.format(based, alg, metric, k,
                                                              fold), log_file)

    file = os.path.join(pred_path, file_name)
    pred_dict = sup.read_cpkl(file)

    sup.logmsg('{}_{}_{}_k{}_f{} - Done reading predict file'.format(based, alg,
                                                                     metric, k, fold),
               log_file)

    coverage = pred_dict.pop(-1)

    mae = 0.0
    mse = 0.0
    n = float(len(pred_dict))

    for key in pred_dict.keys():
        rat, pred = pred_dict[key]
        # sup.logmsg('{}_{}_{}_k{}_f{} - {}'.format(based, alg, metric, k,
        # 	fold, (rat, pred)))

        if pred != 0.0:
            err = rat - pred
            mae += abs(err)
            mse += (err * err)
        else:
            n -= 1.
            # sup.logmsg('{}_{}_{}_c{}_k{}_f{} - ERROR: {} prediction equals 0'.format(
            #     based, alg, metric, k, fold, key), log_file)
        # sup.logmsg('{}_{}_{}_k{}_f{} - {}'.format(based, alg, metric, k,
        # fold, (rmse, mae, mse)))

    mae /= n
    mse /= n
    rmse = sqrt(mse)

    sup.logmsg('{}_{}_{}_k{}_f{} - Writing evaluation on disk'.format(based, alg,
                                                                      metric, k, fold),
               log_file)

    # print (rmse, mse, mae)

    out_file_name = '{}_{}_{}_k{}_f{}.csv'.format(based, alg, metric, k, fold)
    file_path = os.path.join(output_folder, 'knn_evaluation')

    if not os.path.exists(file_path):
        try:
            os.makedirs(file_path)
        except:
            pass

    file = os.path.join(file_path, out_file_name)

    f = open(file, 'w')

    rmse = str(rmse).replace('.', ',')
    mse = str(mse).replace('.', ',')
    mae = str(mae).replace('.', ',')
    coverage = str(coverage).replace('.', ',')

    f.write('RMSE:;{}\n'.format(rmse))
    f.write('MSE:;{}\n'.format(mse))
    f.write('MAE:;{}\n'.format(mae))
    f.write('Coverage:;{}\n'.format(coverage))
    f.write('0 errors:;{}\n\n'.format(str(float(len(pred_dict)) - n).replace('.', ',')))

    f.write('userID; itemID; Rating; Predicted\n')

    for key in pred_dict.keys():
        userId, itemId = key
        rat, pred = pred_dict[key]

        rat = str(rat).replace('.', ',')
        pred = str(pred).replace('.', ',')

        f.write('{};{};{};{}\n'.format(userId, itemId, rat, pred))

    f.close()

    pred_dict.clear()

    sup.logmsg(
        '{}_{}_{}_k{}_f{} - Done Evaluating ratings {}'.format(based, alg, metric, k,
                                                               fold, (rmse, mse, mae,
                                                                      coverage)),
        log_file)

    return rmse, mse, mae, coverage


num_cores = multiprocessing.cpu_count()

data_path = sys.argv[1]
pred_path = os.path.join(data_path, 'knn_predictions')
output_folder = data_path
num_folds = int(sys.argv[2])

NUM_USERS, NUM_ITEMS = cpkl.load(open(os.path.join(data_path, 'dataset_size.pkl'),
                                      'rb'))
k_list = cpkl.load(open(os.path.join(data_path, 'k_list.pkl'), 'rb'))

alg_list = ['brute', 'ball_tree', 'kd_tree']
metric_list = ['cosine', 'euclidean', 'minkowski']

log_file = sup.create_logfile(sys.argv[0])
p_list = []

for based in ['user', 'item']:
    for alg in alg_list:
        for metric in metric_list:
            if metric != 'cosine' or alg == 'brute':
                for k in k_list:
                    for fold in range(num_folds):
                        p_list.append((based, alg, metric, k, fold))

err = Parallel(n_jobs=-1, backend="threading")(delayed(p_loop)(p[0], p[1], p[2], p[3],
                                                               p[4], output_folder,
                                                               pred_path, log_file)
                                               for p in p_list)

# based, alg, metric, k, fold, output_folder, pred_path, num_clusters, log_file


error_dict = {}

for p in p_list:
    key = (p[0], p[1], p[2], p[3], p[4])
    error_dict[key] = err.pop(0)

file = os.path.join(output_folder, 'knn_eval.cpkl')
sup.write_cpkl(error_dict, file)

file = os.path.join(output_folder, 'knn_eval.csv')

f = open(file, 'w')

f.write('based;alg;metric;k;fold;;rmse;mse;mae;coverage\n')

for key in sorted(error_dict.keys()):
    rmse, mse, mae, coverage = error_dict[key]
    based, alg, metric, k, fold = key

    f.write('{};{};{};{};{};;{};{};{};{}\n'.format(based, alg, metric, k,
                                                      fold, rmse, mse, mae, coverage))

f.close()
