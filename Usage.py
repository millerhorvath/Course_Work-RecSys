from subprocess import call
import os
import support_functions as sup


# Path of dataset file
data_path = os.path.join("dataset_small", "u.data")

# Does the dataset have a header line?
has_header = "0"

# Column separator of the dataset file
separator = "\t"

# Path to output processed data
output_path = "data"

# Number of folds for the k-fold cross-validation
folds_number = "5"

# Path for the clustered data
kmeans_path = os.path.join(output_path, 'kmeans')

# Running the 1st script for data sampling
py_file = "01_data_split_function.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, data_path, has_header, separator, output_path])

print ""  # Break line

# Running the 2nd script for clustering computing
py_file = "02_k-means.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path])

print ""  # Break line

# Running the 3rd script for fitting test data into clusters
py_file = "03_finding_clusters.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path, folds_number, kmeans_path])

print ""  # Break line

# Running the 4th set of scripts for computing similarity matrices for all models
py_file = "04_kmeans_similarity_matrices.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path, folds_number])

print ""  # Break line

py_file = "04_knn_similarity_matrices.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path, folds_number])

print ""  # Break line

# Running the 5th set of scripts for predicting ratings for all models
py_file = "05_kmeans_nonzero_predict.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path, folds_number])

print ""  # Break line

py_file = "05_knn_nonzero_predict.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path, folds_number])

print ""  # Break line

# Running the 6th set of scripts for predicting ratings
py_file = "06_kmeans_evaluate_predictions.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path, folds_number])

print ""  # Break line

py_file = "06_knn_evaluate_predictions.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path, folds_number])

print ""  # Break line

# Running the 7th script for computing the linear weighted hybrid coefficients
py_file = "07_compute_evaluate_hybrid.py"
sup.logmsg("Running '{}' script".format(py_file))
call(["python", py_file, output_path, folds_number])

