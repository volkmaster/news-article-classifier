#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'


from os.path import basename
from sys import argv
import time
from Data import Features
from Util import Classifier, ActiveLearning


def main():
    n_inner = 10
    approaches = ['random', 'margin', 'entropy', 'margin-correlation', 'entropy-correlation']

    # optimal combination of parameters for model evaluation
    feature_indices = [1]
    combinations = {'logreg': [100]}

    # load test set features
    features_test = Features(dir_name='wiki_guardian/test', labeled=True)
    features_test.load()

    for n in range(n_inner+1):
        for app in approaches:
            print len(app) * '*', '\n', app, '\n', len(app) * '*'

            # load training set features
            features_train = Features(dir_name='wiki_guardian/%s/train_%d' % (app, n), labeled=True)
            features_train.load()

            # run model evaluation with optimal combination of features, algorithm and parameter
            clf = Classifier(mode='model_evaluation', approach=app, n=n, skewed_class_weighing=False,
                             feature_indices=feature_indices, combinations=combinations, trace=True)
            clf.fit_predict(train=features_train, test=features_test)
            clf.evaluate_and_save_results()

    ActiveLearning.plot_curves(n_inner, feature_indices, combinations, metric='fscore')


if __name__ == '__main__':
    t0 = time.time()
    print
    main()
    print
    print '%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0)
