#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

import os
from os.path import basename
from sys import argv
import time
from mllib import DataLoader
from Data import ppr, WikiOriginal, WikiScraped, WikiJoint, Guardian, GuardianJoint, WikiGuardian
from Util import NELLDict


def main():
    nell = NELLDict(trace=True)
    nell.load()

    #########################
    # ORIGINAL WIKI ARTICLES
    #########################
    wo = WikiOriginal()
    articles_big = DataLoader.load(wo.path + 'articles_big.json')['article-list']
    wo.parse(articles_big)
    wo.preprocess_body_event_type()     # remove HTML tags from body and convert event type to lower
    wo.preprocess_metadata()            # preprocess categories, keywords and entities
    wo.remove_duplicate_metadata()
    nell.update(wo.article_dict)        # update NELL dictionary (query NELL "JSON0" API)
    wo.clean()                          # clean titles and bodies (stem, remove stop words); save to new file

    ########################
    # SCRAPED WIKI ARTICLES
    ########################
    # scrape article URLs and event types from Wikipedia's Current Events Portal
    os.system('scrapy runspider spider_wiki.py -o data/articles/wiki/scraped/urls_event_types.json')

    ws = WikiScraped()
    ws.load('articles.json')
    urls_event_types = DataLoader.load(ws.path + 'urls_event_types.json')
    ws.query_er(urls_event_types)   # add title, body and categories (query EventRegistry)
    ws.remove_duplicates(wo.article_dict)
    ws.remap_event_types()
    ws.add_keywords()
    ws.preprocess_body()            # remove HTML tags from body
    ws.add_entities()               # add entities (query enrycher)
    ws.preprocess_metadata()        # preprocess categories, keywords and entities
    ws.remove_duplicate_metadata()
    nell.update(ws.article_dict)    # update NELL dictionary (query NELL "JSON0" API)
    ws.clean()                      # clean titles and bodies (stem, remove stop words); save to new file

    ######################
    # JOINT WIKI ARTICLES
    ######################
    wo = WikiOriginal()
    ws = WikiScraped()
    w = WikiJoint()

    wo.load('articles.json')
    ws.load('articles.json')
    w.join([wo.article_dict, ws.article_dict], 'articles.json')

    wo.load('articles_preprocessed.json')
    ws.load('articles_preprocessed.json')
    w.join([wo.article_dict, ws.article_dict], 'articles_preprocessed.json')

    #####################
    # GUARDIAN INSTANCES
    #####################
    classes = {
        0: 'armed conflicts and attacks',
        1: 'arts and culture',
        2: 'business and economy',
        3: 'disasters and accidents',
        4: 'law and crime',
        5: 'politics and elections',
        6: 'science and technology',
        7: 'sport'
    }

    article_dict_list, article_dict_preprocessed_list = [], []
    for i in range(len(classes)):
        print 100 * '-', '\n%s\n' % classes[i], 100 * '-'

        section = ['world',
                   'artanddesign|books|culture|culture-network|culture-professionals-network|music',
                   'business|small-business-network',
                   'environment|weather|local|world|news',
                   'lawl|ocal|news|us-news|uk-news',
                   'politics|public-leaders-network|local-leaders-network|us-news|uk-news|world',
                   'science|technology',
                   'sport']
        content = ['(("armed conflicts") OR (terrorism AND suicide)) AND war',
                   '',
                   '',
                   '(disasters OR accidents OR death) AND NOT law',
                   '(law AND legislation) OR crime OR murder OR kill OR death AND NOT (war OR terrorism)',
                   'politics OR election OR senate OR negotiations OR agreement OR "international relations"',
                   'science OR technology OR resarch OR discovery OR medicine',
                   '']
        gi = Guardian(event_type=classes[i], n_pages=6)
        gi.query(section[i], content[i])           # add title, body and keywords (query Guardian API)

        # debug
        # articles = DataLoader.load(gi.path + '0.json' % classes[i])
        # ppr(articles[0])

        gi.parse_add_entities()                    # remove HTML tags from body, add entities (query enrycher)
        gi.load('articles.json')                   # reason: unicode problem
        gi.preprocess_metadata()                   # preprocess categories, keywords and entities
        gi.remove_duplicate_metadata()
        article_dict_list.append(gi.article_dict)
        nell.update(gi.article_dict)               # update NELL dictionary (query NELL "JSON0" API)
        gi.clean()                                 # clean titles and bodies (stem, remove stop words); save to new file
        article_dict_preprocessed_list.append(gi.article_dict)

    ##########################
    # JOINT GUARDIAN ARTICLES
    ##########################
    g = GuardianJoint()
    g.join(article_dict_list, 'articles.json')
    g.join(article_dict_preprocessed_list, 'articles_preprocessed.json')

    #################
    # JOINT ARTICLES
    #################
    w = WikiJoint()
    w.load('articles.json')
    g = GuardianJoint()
    g.load('articles.json')
    a = WikiGuardian()
    a.join([w.article_dict, g.article_dict], 'articles.json')

    w = WikiJoint()
    w.load('articles_preprocessed.json')
    g = GuardianJoint()
    g.load('articles_preprocessed.json')
    a = WikiGuardian()
    a.join([w.article_dict, g.article_dict], 'articles_preprocessed.json')

    ##########################
    # EVENT TYPE DISTRIBUTION
    ##########################
    a = WikiGuardian()
    a.load('articles_preprocessed.json')
    a.display_event_type_distribution()


if __name__ == '__main__':
    t0 = time.time()
    print
    main()
    print
    print '%s executed successfully in %.2f s.' % (basename(argv[0]), time.time() - t0)
