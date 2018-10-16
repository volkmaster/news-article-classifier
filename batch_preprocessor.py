#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

from os.path import basename
from sys import argv
import time
from Util import Logger, Config, Batch, NELLDict, FeaturePreprocessor

t0 = time.time()

logger = Logger(type_='bp')

config = Config(logger=logger)
config.load()

nell = NELLDict(logger=logger, trace=True)
nell.load()

batch = Batch(logger=logger, n=config.iter_bp)
batch.clean()
batch.load()

nell.update(batch.article_dict)

features_batch = FeaturePreprocessor(logger=logger, type_='batch', n=config.iter_bp)
features_batch.transform(batch.article_dict, nell.entity_categories)

config.stop('bp')       # indicate that the batch preprocessor is finished

logger('%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0))
