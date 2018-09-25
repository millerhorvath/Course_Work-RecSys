import os
import cPickle as cpkl
import sys
import datetime
import support_functions as sup


def logmsg(*args):
    print '[', datetime.datetime.now(), ']', ' '.join([str(a) for a in args])


file_path = sys.argv[1]
has_header = int(sys.argv[2])
separator = sys.argv[3]
output_folder = sys.argv[4]

# # # # # MODIFY THE FOLLOWING 2 LINES TO CUSTOMIZE THE EXPERIMENT PARAMETERS
cluster_list = [2, 5]  # List of cluster parameters (for both users and items)
k_list = [50, 100]  # List of neighborhood (k) parameters (for kNN)

# Storing the list of parameters for further scripts
sup.write_cpkl2(cluster_list, output_folder, 'cluster_list.pkl')
sup.write_cpkl2(k_list, output_folder, 'k_list.pkl')

logmsg('Starting creating dataset dictonaries')

# def data_split(folder_path, file_name, has_reader):
test_path = os.path.join(output_folder, 'test')
train_path = os.path.join(output_folder, 'train')
dict_path = os.path.join(output_folder, 'dicts')

f = open(file_path, 'r')

if has_header:
    logmsg('The dataset file has a header line')
    header = f.readline()

data = f.readlines()

f.close()

UsrItemRat = {}
userId_dict = {}
itemId_dict = {}
inv_userId_dict = {}
inv_itemId_dict = {}

# lastUser = -1

for line in data:
    d = line.split(separator)

    if d[0] not in userId_dict:
        userId_dict[d[0]] = len(userId_dict)
        inv_userId_dict[len(userId_dict) - 1] = d[0]
        UsrItemRat[userId_dict[d[0]]] = []

    if d[1] not in itemId_dict:
        itemId_dict[d[1]] = len(itemId_dict)
        inv_itemId_dict[len(itemId_dict) - 1] = d[1]

    userId = userId_dict[d[0]]
    itemId = itemId_dict[d[1]]

    UsrItemRat[userId].append([itemId, d[2]])

dataset_size = len(userId_dict), len(itemId_dict)

logmsg('Dictionaries created!')

if not os.path.exists(dict_path):
    os.makedirs(dict_path)

logmsg('Writing dictonaries on disk...')

cpkl.dump(userId_dict, open(os.path.join(dict_path, 'userId_dict.pkl'), 'wb'))
cpkl.dump(inv_userId_dict, open(os.path.join(dict_path, 'inv_userId_dict.pkl'), 'wb'))
cpkl.dump(itemId_dict, open(os.path.join(dict_path, 'itemId_dict.pkl'), 'wb'))
cpkl.dump(inv_itemId_dict, open(os.path.join(dict_path, 'inv_itemId_dict.pkl'), 'wb'))
cpkl.dump(dataset_size, open(os.path.join(output_folder, 'dataset_size.pkl'), 'wb'))

logmsg('Starting spliting the dataset')

test = []
train = []

for val in range(5):
    test.append([])
    train.append([])

    i = 0

    for i in sorted(UsrItemRat.keys()):
        film = 0
        for j in UsrItemRat[i]:
            if film != val:
                train[val].append([str(i), str(j[0]), j[1]])
            else:
                test[val].append([str(i), str(j[0]), j[1]])
            film = (film + 1) % 5

# print (test[0])
# print (train[0])

logmsg('Dataset split')
logmsg('Writing test datasets on disk...')

for val in range(5):
    write_path = os.path.join(test_path, "myTest{}.csv".format(val))

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    f = open(write_path, "w")
    f.write("userId,movieId,rating\n")

    for i in test[val]:
        f.write(','.join(i))
        f.write('\n')

    f.close()

logmsg('Writing training datasets on disk...')

for val in range(5):
    write_path = os.path.join(train_path, "myTrain{}.csv".format(val))

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    f = open(write_path, "w")
    f.write("userId,movieId,rating\n")

    for i in train[val]:
        i[0] = '{}'.format(i[0])
        f.write(','.join(i))
        f.write('\n')

    f.close()

logmsg('Spliting data done!')
