from __future__ import absolute_import, division, print_function

import os
import pickle
import gzip

from utils import *
from data_utils import AmazonDataset
from knowledge_graph import KnowledgeGraph


def generate_train_labels(dataset):
    train_review_file = DATASET_DIR[dataset] + '/query_split/train.txt.gz'
    user_products = {}  # {uid: [pid,...], ...}
    with gzip.open(train_review_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    save_labels(dataset, user_products, mode='train')


def generate_test_labels(dataset):
    test_review_file = DATASET_DIR[dataset] + '/query_split/test.txt.gz'
    user_products = {}  # {uid: [pid,...], ...}
    with gzip.open(test_review_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    save_labels(dataset, user_products, mode='test')


def main():
    datasets = [CELL, BEAUTY, CLOTH, CD]

    # Create AmazonDataset instances for each dataset.
    # ========== BEGIN ========== #
    for name in datasets:
        print('Load', name, 'dataset from file...')
        dataset = AmazonDataset(DATASET_DIR[name])
        if not os.path.isdir(TMP_DIR[name]):
            os.makedirs(TMP_DIR[name])
        save_dataset(name, dataset)
    # =========== END =========== #

    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    for name in datasets:
        print('Create', name, 'knowledge graph from dataset...')
        dataset = load_dataset(name)
        kg = KnowledgeGraph(dataset)
        kg.compute_degrees()
        save_kg(name, kg)
    # =========== END =========== #

    # Genereate train/test labels.
    for name in datasets:
        print('Generate', name, 'train/test labels.')
        generate_train_labels(name)
        generate_test_labels(name)


if __name__ == '__main__':
    main()
