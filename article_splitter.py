#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

import os
from os.path import basename
from sys import argv
import time
import numpy as np
from Data import WikiGuardian, Features
from Util import NELLDict


seed1 = 1813
seed2 = 0


def main():
    # load articles
    wg = WikiGuardian()
    wg.load('articles_preprocessed.json')

    # shuffle article ids of whole article set
    article_ids = np.array(wg.article_dict.keys())
    order = np.arange(len(article_ids))
    np.random.seed(seed1)
    np.random.shuffle(order)
    article_ids = article_ids[order]

    # select stratified subset (small no. of articles per event type)
    selected = wg.select_stratified_subset(shuffled_article_ids=article_ids, n_per_event_type=1250)
    article_ids = np.array(sum(selected.values(), []))
    order = np.arange(len(article_ids))
    np.random.shuffle(order)
    article_ids = article_ids[order]

    # divide article ids
    n_train = int(round(len(article_ids) * 0.1))
    n_test = int(round(len(article_ids) * 0.1))
    train_ids = article_ids[:n_train]
    test_ids = article_ids[n_train:n_train+n_test]
    active_learning_ids = article_ids[n_train+n_test:]

    # shuffle active learning set ids
    order = np.arange(len(active_learning_ids))
    np.random.seed(seed2)
    np.random.shuffle(order)
    active_learning_ids = active_learning_ids[order]
    n_inner = 10
    n_per_iter = len(active_learning_ids) / n_inner

    approaches = ['random', 'margin', 'entropy', 'margin-correlation', 'entropy-correlation']

    # split and save articles into 3 subsets
    for app in approaches:
        wg.select_subset(train_ids, 'split/%s/train_0.json' % app, save=True)
    for n in range(n_inner):
        wg.select_subset(active_learning_ids[n*n_per_iter:(n+1)*n_per_iter], 'split/active_learning_%d.json' % n, save=True)
    wg.select_subset(test_ids, 'split/test.json', save=True)

    # load NELL dict
    nell = NELLDict()
    nell.load()

    # load initial training and create features
    for app in approaches:
        wg.load('split/%s/train_0.json' % app)
        features_train = Features(dir_name='wiki_guardian/%s/train_0' % app, labeled=True)
        features_train.transform(wg.article_dict, nell.entity_categories)

    # load active learning sets and create features
    for n in range(n_inner):
        wg.load('split/active_learning_%d.json' % n)
        features_active_learning = Features(dir_name='wiki_guardian/active_learning_%d' % n, labeled=True)
        features_active_learning.transform(wg.article_dict, nell.entity_categories)

    # load test set and create features
    wg.load('split/test.json')
    features_test = Features(dir_name='wiki_guardian/test', labeled=True)
    features_test.transform(wg.article_dict, nell.entity_categories)


if __name__ == '__main__':
    t0 = time.time()
    print
    main()
    print
    print '%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0)
    # os.system('python grid_search.py')
    os.system('python active_learning.py ' + str(seed2))
