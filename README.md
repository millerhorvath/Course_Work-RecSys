modules needed: joblib, sklearn and numpy

1 - Introduction

This folder contains the python scripts used to run the experiments of my coursework at
Centro Universitario FEI.
The work aims at exploting cluster especialization in hybrid recommender systems.
Because of the coursework deadline, we only adopted user-based and item-based
collaborative filtering algoritms for recommendation and k-Means algorithm for
clustering.
Future research will explore other state-of-art recommender and clustering algorithms.
The "coursework_diagram.pdf" presents a diagram summarizing my coursework methodology.

2 - Dataset

The original dataset used to evaluate the proposed methodology was 1M MovieLens
Dataset ("dataset" folder).
Due to computational time, the "Usage.py" is modeled to use a smaller dataset, the
100k MovieLens Dataset ("dataset_small" folder).
More info about the datasets at: https://grouplens.org/datasets/movielens/

3 - Usage

To run the entire experiment, just run the "Usage.py" script.
There you can find the order the scrips should be run and their respective arguments.
Although the scripts were writen using parallel processing in order to exploit all the
computational resource avaialble (multithread based, not cluster based), some scripts
might take a relatively long time to run due to the dataset size and disk I/O
opperations.
Moreover, a considerable amount of data storage is needed (about 2.1 GB for the default experiment
configuration).

To adapt this code to different rating-based recommender datasets (user-id, item-id,
rating), you have to modify the "01_data_split_function.py" script in order to read and
sample your dataset properly and configure the experiments parameters as you please
(basic python knowledge is needed).

Furthermore, you might want to analyse the results generated after the standalone
models evaluation in order to define the hybrid model parameters based on the
standalone results.
To do so, you have to modify the "07_compute_evaluate_hybrid.py" (basic python
knowledge is needed).









