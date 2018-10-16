#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

from os.path import basename
from sys import argv
import time
from Util import Logger, Config, FeaturePreprocessor, Classifier

t0 = time.time()
logger = Logger(type_='cv')

config = Config(logger=logger)
config.load()

features_batch = FeaturePreprocessor(logger=logger, type_='batch', n=config.iter_cv)
features_batch.load()

clf = Classifier(logger=logger, n=config.iter_cv, idx=config.idx, cv=True, c=config.c, shuffle=True, trace=True)
clf.fit_predict(train=features_batch.features)

config.increment_iter('cv')     # increment the cross-validation iteration no.
config.stop('cv')               # indicate that the cross-validation is finished

logger('%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0))
