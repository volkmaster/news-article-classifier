#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'


from os.path import basename
from sys import argv
import time
from Data import Features
from Util import Classifier


def main():
    # load training set features
    features_train = Features(dir_name='wiki_guardian/random/train_0', labeled=True)
    features_train.load()

    # define combinations for parameter optimization grid search
    feature_indices = [0, 1, 2, 3, 4, 5]
    combinations = {'knn': [11, 19, 27], 'rf': [10, 50, 100], 'svm': [1, 10, 100], 'logreg': [1, 10, 100]}

    # run grid search using k-fold cross-validation with different feature, algorithm and parameter combinations
    clf = Classifier(mode='cv', k=10, skewed_class_weighing=False, feature_indices=feature_indices,
                     combinations=combinations, trace=True)
    clf.fit_predict(train=features_train)
    clf.evaluate_and_save_results()


if __name__ == '__main__':
    t0 = time.time()
    print
    main()
    print
    print '%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0)
