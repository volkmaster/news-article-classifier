#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'


import os
from os.path import basename
from sys import argv
import time
from Data import WikiGuardian, Features
from Util import NELLDict, ActiveLearning


seed = int(argv[1])


def main():
    n_inner = 10
    n_subset = 400
    approaches = ['random', 'margin', 'entropy', 'margin-correlation', 'entropy-correlation']

    # optimal combination of parameters for active learning uncertainty-based instance selection
    feature_indices = [1]
    combinations = {'logreg': [100]}

    # load NELL dict
    nell = NELLDict()
    nell.load()

    for n in range(n_inner):
        # load active learning set articles and features
        wg_active_learning = WikiGuardian()
        wg_active_learning.load('split/active_learning_%d.json' % n)
        features_active_learning = Features(dir_name='wiki_guardian/active_learning_%d' % n, labeled=True)
        features_active_learning.load()

        for app in approaches:
            print len(app) * '*', '\n', app, '\n', len(app) * '*'

            # load training set articles and features
            wg_train = WikiGuardian()
            wg_train.load('split/%s/train_%d.json' % (app, n))
            features_train = Features(dir_name='wiki_guardian/%s/train_%d' % (app, n), labeled=True)
            features_train.load()

            # select a subset of articles according to the active learning instance selection approach
            al = ActiveLearning(n=n, approach=app, n_subset=n_subset,
                                features_train=features_train, features_active_learning=features_active_learning,
                                feature_indices=feature_indices, combinations=combinations, seed=seed, beta=2)
            article_ids = al.select()

            # select the subset from active learning set of articles and add it to the training set of articles
            article_dict = wg_active_learning.select_subset(article_ids, save=False)
            wg_train.join([wg_train.article_dict, article_dict], 'split/%s/train_%d.json' % (app, n+1))

            # create new training set features
            features_train = Features(dir_name='wiki_guardian/%s/train_%d' % (app, n+1), labeled=True)
            features_train.transform(wg_train.article_dict, nell.entity_categories)


if __name__ == '__main__':
    t0 = time.time()
    print
    main()
    print
    print '%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0)
    os.system('python model_evaluation.py')
