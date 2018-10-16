#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

from os.path import basename
from sys import argv
import time
from Util import Logger, Config, Streamer, NewsStream, NELLDict, FeaturePreprocessor

t0 = time.time()

logger = Logger(type_='nsp')

config = Config(logger=logger)
config.load()

nell = NELLDict(logger=logger, trace=True)
nell.load()

Streamer(logger=logger, n=config.iter_nsp).fetch()

news_stream = NewsStream(logger=logger, n=config.iter_nsp)
news_stream.clean()
news_stream.load()

nell.update(news_stream.article_dict)

features_news_stream = FeaturePreprocessor(logger=logger, type_='news_stream', n=config.iter_nsp)
features_news_stream.transform(news_stream.article_dict, nell.entity_categories)

config.increment_iter('nsp')    # increment the news stream preprocessor iteration no.
config.stop('nsp')              # indicate that the news stream preprocessor is finished

logger('%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0))
