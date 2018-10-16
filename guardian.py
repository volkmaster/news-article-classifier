#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

from collections import Counter
import pprint
from unidecode import unidecode
import xml.etree.cElementTree as ET
from HTMLParser import HTMLParser
from nltk import PorterStemmer
from re import sub
import numpy as np
import matplotlib.pyplot as plt
import requests
from EventRegistry import *
from mllib import DataLoader, DataSaver


class GuardianInstance:
    #################
    # global
    #################

    def __init__(self, path='data/articles/guardian/', event_type=None, n_pages=0):
        self.path = path
        self.event_type = event_type
        self.n_pages = n_pages
        self.article_dict = {}

    def __contains__(self, article_id):
        return article_id in self.article_dict

    def __call__(self, article_id):
        msg = ''
        if article_id in self:
            a = self.article_dict[article_id]
            msg += 'Article ID:\t\t\t%s\n' % article_id
            msg += 'URL:\t\t\t\t%s\n' % a.url
            msg += 'Title:\t\t\t\t%s\n' % a.title
            msg += 'Body:\t\t\t\t%s\n' % a.body
            msg += 'Keywords:\t\t%s\n' % str(a.keywords)
            msg += 'Entities:\t\t\t%s\n' % str(a.entities)
            msg += 'Event type:\t\t\t%s' % a.event_type
        else:
            msg = 'No article with such ID.'

        return msg

    class _Article:
        def __init__(self):
            self.url = ''
            self.title = ''
            self.body = ''
            self.categories = []
            self.keywords = []
            self.entities = []
            self.event_type = ''

    #################
    # main functions
    #################

    def query(self, section, content):
        url = 'http://content.guardianapis.com/search'
        for page in range(1, self.n_pages+1):
            params = {
                'section': section,
                'page-size': 200,
                'page': page,
                'show-fields': 'body,headline',
                'show-tags': 'keyword',
                'order-by': 'relevance',
                'q': content,
                'api-key': 'akws72hy56nyh8xhnuasf4ax'
            }
            response = requests.get(url, params)
            data = json.loads(response.text)['response']['results']

            # debug
            # self._ppr(data[0])

            DataSaver.save(self.path + '%s/%d.json' % (self.event_type, page-1), data)

    def parse_add_entities(self):
        if os.path.exists(self.path + '%s/articles.json' % self.event_type):
            self.article_dict = self.load('%s/articles.json' % self.event_type)
        else:
            self.article_dict = {}

        for page in range(self.n_pages):
            articles = DataLoader.load(self.path + '%s/%d.json' % (self.event_type, page))

            # debug
            # self._ppr(articles[0])

            for i, a in enumerate(articles):
                # intermediate save
                if i > 0 and i % 50 == 0:
                    self._save('%s/articles.json' % self.event_type, self.article_dict, 'parsed and enriched with entities')

                try:
                    body = a['fields']['body']
                    if a['id'] not in self.article_dict and a['type'] == 'article':
                        article = self._Article()
                        article.url = a['webUrl']
                        article.title = a['fields']['headline']
                        article.body = self._remove_html_tags(body)
                        response = self._enrych(article.body)
                        try:
                            root = ET.fromstring(response)
                            for annotation in root.find('annotations').findall('annotation'):
                                article.entities.append(annotation.attrib['displayName'])
                            print '# %d = %d entities' % ((page * 200) + i, len(article.entities))
                        except ET.ParseError as e:
                            print '# %d = %s' % ((page * 200) + i, e)
                            continue
                        article.keywords = [k['webTitle']for k in a['tags']]
                        article.event_type = self.event_type
                        self.article_dict[a['id']] = article
                except KeyError:
                    continue    # no body present

            # intermediate save
            self._save('%s/articles.json' % self.event_type, self.article_dict, 'parsed and enriched with entities')

        print 'Average no. of entities per article:', np.mean([len(a.entities) for _, a in self.article_dict.iteritems()])

        # final save
        self._save('%s/articles.json' % self.event_type, self.article_dict, 'parsed and enriched with entities')

    def preprocess_metadata(self):
        h = HTMLParser()
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # convert unicode characters to ascii, convert to lower and replace underscores
            a.categories = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.categories]
            a.keywords = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.keywords]
            a.entities = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.entities]

        self._save('%s/articles.json' % self.event_type, self.article_dict, 'preprocessed (metadata)')

    def remove_duplicate_metadata(self):
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            cat1.append(len(a.categories))
            key1.append(len(a.keywords))
            ent1.append(len(a.entities))

            # uniquify categories, keywords and entities
            a.categories = [val for val in list(set(a.categories)) if (val not in a.keywords) and (val not in a.entities)]
            a.keywords = [val for val in list(set(a.keywords)) if (val not in a.categories) and (val not in a.entities)]
            a.entities = [val for val in list(set(a.entities)) if (val not in a.categories) and (val not in a.keywords)]

        self._save('%s/articles.json' % self.event_type, self.article_dict, 'uniquified (metadata)')

    def clean(self):
        # initialize Porter stemmer and load stop words
        stemmer = PorterStemmer()
        stop_words = [word.decode('utf-8') for word in DataLoader.load('data/stop_words/stop_words.txt', dtype=str)]

        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # convert characters to lower
            a.title = a.title.lower()
            a.body = a.body.lower()

            # replace new line characters with spaces
            a.body = a.body.replace('\n', ' ')

            # remove numbers and punctuations
            a.title = sub(r'[^a-z ]', '', a.title)
            a.body = sub(r'[^a-z ]', '', a.body)

            # stem and skip stop words
            title = []
            for word in a.title.select_subset():
                word = stemmer.stem_word(word)
                if word not in stop_words:
                    title.append(word)
            a.title = ' '.join(title)

            body = []
            for word in a.body.select_subset():
                word = stemmer.stem_word(word)
                if word not in stop_words:
                    body.append(word)
            a.body = ' '.join(body)

        self._save('%s/articles_preprocessed.json' % self.event_type, self.article_dict, 'cleaned (stemmed, removed stop words)')

    def count_unique_urls(self, event_type, n_pages):
        urls = []
        for i in range(n_pages):
            articles = DataLoader.load(self.path + '%s/guardian_%d.json' % (event_type, i))
            urls += [article['webUrl'] for article in articles]
        print 'Unique URLs in Guardian %s article set: %d / %d' % (event_type, len(set(urls)), n_pages*200)

    def display_event_type_distribution(self):
        counts = Counter([article.event_type for _, article in self.article_dict.iteritems()])
        for event_type, count in counts.iteritems():
            print '%s: %.1f %%' % (event_type, (count * 100.) / len(self.article_dict))

        fig, ax = plt.subplots()
        ind = np.arange(len(counts))
        width = 0.5
        ax.bar(ind, counts.values(), width=width, color='g', alpha=0.6)
        ax.set_ylabel('Article count')
        ax.set_title('Event type support')
        ax.set_xticks(ind + 0.1)
        xtick_names = ax.set_xticklabels([event_type[:10] + '..' for event_type in counts.keys()])
        plt.setp(xtick_names, rotation=30, fontsize=9)
        ax.set_xlim([-0.5, len(counts)])
        ax.grid(True)
        plt.show()

    ####################
    # utility functions
    ####################

    @staticmethod
    def _ppr(x):
        pprint.PrettyPrinter(indent=4).pprint(x)

    def _save(self, file_name, article_dict, desc):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Guardian article set (%d articles) %s and saved to %s.' % (len(article_dict), desc, path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Guardian article set (%d articles) loaded from %s.' % (len(articles), path)

        # debug
        # self._ppr(articles[articles.keys()[0]])

        for article_id, a in articles.iteritems():
            article = self._Article()

            article.url = a['url']
            article.title = a['title']
            article.body = a['body']
            article.entities = a['entities']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict

    @staticmethod
    def _enrych(text):
        data = ('<item><text>' + text.encode('utf-8') + '</text></item>')
        req = urllib2.Request(url='http://mustang.ijs.si:4091/annotation-en/ngram',
                              data=data,
                              headers={'Content-type': 'text/xml'})
        response = urllib2.urlopen(req, timeout=60*30).read()

        return response

    @staticmethod
    def _remove_html_tags(text):
        class _HTMLTagStripper(HTMLParser):
            def __init__(self):
                self.reset()
                self.fed = []

            def handle_data(self, d):
                self.fed.append(d)

            def get_data(self):
                return ''.join(self.fed)
        s = _HTMLTagStripper()
        s.feed(text)
        return s.get_data()


class Guardian:
    #################
    # global
    #################

    def __init__(self, path='data/articles/guardian/'):
        self.path = path
        self.article_dict = {}

    def __contains__(self, article_id):
        return article_id in self.article_dict

    def __call__(self, article_id):
        msg = ''
        if article_id in self:
            a = self.article_dict[article_id]
            msg += 'Article ID:\t\t\t%s\n' % article_id
            msg += 'URL:\t\t\t\t%s\n' % a.url
            msg += 'Title:\t\t\t\t%s\n' % a.title
            msg += 'Body:\t\t\t\t%s\n' % a.body
            msg += 'Keywords:\t\t%s\n' % str(a.keywords)
            msg += 'Entities:\t\t\t%s\n' % str(a.entities)
            msg += 'Event type:\t\t\t%s' % a.event_type
        else:
            msg = 'No article with such ID.'

        return msg

    class _Article:
        def __init__(self):
            self.url = ''
            self.title = ''
            self.body = ''
            self.categories = []
            self.keywords = []
            self.entities = []
            self.event_type = ''

    #################
    # main functions
    #################
    def join_articles(self, article_dict_list, file_name):
        self.article_dict = article_dict_list[0]
        for i in range(1, len(article_dict_list)):
            for article_id, article in article_dict_list[i].iteritems():
                self.article_dict[article_id] = article

        self._save(file_name, self.article_dict)

    def display_event_type_distribution(self):
        counts = Counter([article.event_type for _, article in self.article_dict.iteritems()])
        for event_type, count in counts.iteritems():
            print '%s: %.1f %%' % (event_type, (count * 100.) / len(self.article_dict))

        fig, ax = plt.subplots()
        ind = np.arange(len(counts))
        width = 0.5
        ax.bar(ind, counts.values(), width=width, color='g', alpha=0.6)
        ax.set_ylabel('Article count')
        ax.set_title('Event type support')
        ax.set_xticks(ind + 0.1)
        xtick_names = ax.set_xticklabels([event_type[:10] + '..' for event_type in counts.keys()])
        plt.setp(xtick_names, rotation=30, fontsize=9)
        ax.set_xlim([-0.5, len(counts)])
        ax.grid(True)
        plt.show()

    ####################
    # utility functions
    ####################

    @staticmethod
    def _ppr(x):
        pprint.PrettyPrinter(indent=4).pprint(x)

    def _save(self, file_name, article_dict):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Joint Guardian article set (%d articles) saved to %s.' % (len(article_dict), path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Joint Guardian article set (%d articles) loaded from %s.' % (len(articles), path)

        # debug
        # self._ppr(articles[articles.keys()[0]])

        for article_id, a in articles.iteritems():
            article = self._Article()

            article.url = a['url']
            article.title = a['title']
            article.body = a['body']
            article.entities = a['entities']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict
