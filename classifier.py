#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

from os.path import basename
from sys import argv
import time
from Util import Logger, Config, FeaturePreprocessor, Classifier, ActiveLearning

t0 = time.time()
logger = Logger(type_='al')

logger('%s run.' % basename(argv[0]))

config = Config(logger=logger)
config.load()

features_batch = FeaturePreprocessor(logger=logger, type_='batch', n=config.iter_al-1)
features_batch.load()
features_news_stream = FeaturePreprocessor(logger=logger, type_='news_stream', n=config.iter_al)
features_news_stream.load()

clf = Classifier(logger=logger, n=config.iter_al, idx=config.idx, c=config.c)
clf.fit_predict(train=features_batch.features, test=features_news_stream.features)

active_learning = ActiveLearning(logger=logger, n=config.iter_al, n_indefinite_per_class=50)
active_learning.select_indefinite_predictions()

config.unpause()    # unpause the active learning loop for user evaluation

logger('%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0))
