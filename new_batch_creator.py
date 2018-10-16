#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

import os
from os.path import basename
from sys import argv
import time
from Util import Logger, Config, Batch, NewsStream, NELLDict, FeaturePreprocessor

t0 = time.time()
logger = Logger(type_='al')

config = Config(logger=logger)
config.load()

nell = NELLDict(logger=logger)
nell.load()

batch = Batch(logger=logger, n=config.iter_al-1)
batch.load()
news_stream = NewsStream(logger=logger, n=config.iter_al)
news_stream.load()

new_batch = Batch(logger=logger, n=config.iter_al)
new_batch.create_new(batch.article_dict, news_stream.article_dict)

features_new_batch = FeaturePreprocessor(logger=logger, type_='batch', n=config.iter_al)
features_new_batch.transform(new_batch.article_dict, nell.entity_categories)

config.increment_iter('al')         # increment the active learning iteration no.

logger('%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0))

os.system('python classifier.py')   # run classifier script to build an enriched classification model
