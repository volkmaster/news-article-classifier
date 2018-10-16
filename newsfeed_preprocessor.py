#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

from os.path import basename
from sys import argv
import time
from Data import NewsFeed, Features
from Util import NELLDict


def main():
    i = 0

    nf = NewsFeed(n=i)
    nf.fetch_from_newsfeed()
    nf.parse()
    nf.remove_duplicates()
    nell = NELLDict(trace=True)
    nell.load()
    nell.update(nf.article_dict)
    nf.summarize()
    nf.clean()

    nf.load()
    features = Features(dir_name='newsfeed/%d' % i, labeled=False)
    features.transform(nf.article_dict, nell.entity_categories)


if __name__ == '__main__':
    t0 = time.time()
    print
    main()
    print
    print '%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0)
