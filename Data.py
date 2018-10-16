#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'

from os import listdir, makedirs
import errno
from collections import defaultdict, Counter
import pprint
from unidecode import unidecode
import xml.etree.cElementTree as ET
from HTMLParser import HTMLParser
from nltk import PorterStemmer
from re import sub, search, match
import traceback
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
import requests
from EventRegistry import *
from mllib import DataLoader, DataSaver
from Util import Summarizer


####################
# utility functions
####################
def ppr(x):
    pprint.PrettyPrinter(indent=4).pprint(x)


def make_dirs(path):
    try:
        makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    
class HTMLTagStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)    

    
def remove_html_tags(text):
    s = HTMLTagStripper()
    s.feed(text)
    return s.get_data()


def enrych(text):
    data = ('<item><text>' + text.encode('utf-8') + '</text></item>')
    req = urllib2.Request(url='http://mustang.ijs.si:4091/annotation-en/ngram',
                          data=data,
                          headers={'Content-type': 'text/xml'})
    response = urllib2.urlopen(req, timeout=60*30).read()

    return response


###############
# main classes
###############
class WikiOriginal:
    #########
    # global
    #########
    def __init__(self, path='data/articles/wiki_guardian/wiki/original/'):
        self.path = path
        make_dirs(path)
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
            msg += 'Categories:\t%s\n' % str(a.categories)
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
    def parse(self, articles):
        for a in articles:
            # debug
            # ppr(a)

            event_type = a['event_type']

            # skip examples whose classes are underrepresented
            if event_type in ['Armed conflicts and attacks', 'Arts and culture', 'Business and economy',
                              'Disasters and accidents', 'Law and crime', 'Politics and elections',
                              'Science and technology', 'Sport']:
                article = self._Article()

                article.url = a['uri']

                if type(a['title']) is unicode:
                    article.title = a['title']

                article.body = a['body-cleartext'].replace('<p>', ' ').replace('</p>', ' ')

                if 'dmoz' in a:
                    for category in a['dmoz']['categories']:
                        if type(category) is unicode:
                            article.categories.append(category.replace('_', ' '))

                    for keyword in a['dmoz']['keywords']:
                        if type(keyword) is unicode:
                            article.keywords.append(keyword.replace('_', ' '))

                if 'annotations' in a:
                    for annotation in a['annotations']:
                        article.entities.append(annotation['displayName'])

                article.event_type = a['event_type']

                self.article_dict[a['id']] = article

        self._save('articles.json', self.article_dict, 'parsed')

    def preprocess_body_event_type(self):
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # remove html markup
            a.body = remove_html_tags(a.body)

            # convert to lower
            a.event_type = a.event_type.lower()

        self._save('articles.json', self.article_dict, 'preprocessed (body, event_type)')

    def preprocess_metadata(self):
        h = HTMLParser()
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # convert unicode characters to ascii, convert to lower and replace underscores
            a.categories = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.categories]
            a.keywords = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.keywords]
            a.entities = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.entities]

        self._save('articles.json', self.article_dict, 'preprocessed (metadata)')

    def remove_duplicate_metadata(self):
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # uniquify categories, keywords and entities
            a.categories = [val for val in list(set(a.categories)) if (val not in a.keywords) and (val not in a.entities)]
            a.keywords = [val for val in list(set(a.keywords)) if (val not in a.categories) and (val not in a.entities)]
            a.entities = [val for val in list(set(a.entities)) if (val not in a.categories) and (val not in a.keywords)]

        self._save('articles.json', self.article_dict, 'uniquified (metadata)')

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

        self._save('articles_preprocessed.json', self.article_dict, 'cleaned (stemmed, removed stop words)')

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
    def _save(self, file_name, article_dict, desc):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Original Wiki article set (%d articles) %s and saved to %s.' % (len(article_dict), desc, path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Original Wiki article set (%d articles) loaded from %s.' % (len(articles), path)

        # debug
        # ppr(articles[articles.keys()[0]])

        self.article_dict = {}
        for article_id, a in articles.iteritems():
            article = self._Article()

            article.url = a['url']
            article.title = a['title']
            article.body = a['body']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.entities = a['entities']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict


class WikiScraped:
    #########
    # global
    #########
    def __init__(self, path='data/articles/wiki_guardian/wiki/scraped/'):
        self.path = path
        make_dirs(path)
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
            msg += 'Categories:\t%s\n' % str(a.categories)
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
    def query_er(self, urls_event_types):
        er = EventRegistry()

        urls_event_types = {unidecode(article['url']): article['event_type'] for article in urls_event_types}
        urls = urls_event_types.keys()

        n_duplicates = 0
        n_non_english = 0

        i = 0
        request_size = 50

        while i < len(urls):
            q = QueryArticle.queryByUrl(urls[i:i+request_size])
            q.addRequestedResult(RequestArticleInfo(includeArticleCategories=True))
            response = er.execQuery(q)

            # debug
            # ppr(response)

            if response:
                for article_id in response:
                    # debug
                    # ppr(response[article_id])

                    a = response[article_id]['info']

                    # skip duplicates
                    if a['isDuplicate']:
                        n_duplicates += 1
                        continue

                    # skip non-english articles
                    elif a['lang'] != 'eng':
                        n_non_english += 1
                        continue

                    # add articles present in ER to article dict
                    else:
                        article = self._Article()
                        article.url = a['url']
                        article.title = a['title']
                        article.body = a['body']
                        article.categories = [c['label'] for c in a['categories']]
                        article.event_type = urls_event_types[article.url]
                        self.article_dict[a['uri']] = article

            i += request_size

            # debug
            print '%d / %d' % (len(self.article_dict), i)

        print 'Wiki articles present in EventRegistry: %d/%d, Duplicates: %d, Non-English: %d' % \
              (len(self.article_dict) + n_duplicates + n_non_english, len(urls_event_types), n_duplicates, n_non_english)
        self._save('articles.json', self.article_dict, 'enriched with EventRegistry data')

    def remove_duplicates(self, article_dict_original):
        n_duplicates = 0

        article_ids = self.article_dict.keys()
        for article_id in article_ids:
            if article_id in article_dict_original:
                del self.article_dict[article_id]
                n_duplicates += 1

        print 'Removed %d duplicates from scraped Wiki article set.' % n_duplicates
        self._save('articles.json', self.article_dict, 'uniquified')

    def remap_event_types(self):
        # debug
        # for event_type, count in Counter([a.event_type for _, a in article_dict.iteritems()]).iteritems():
        #     print event_type, count

        map_event_type = {
            # 'armed conflicts and attacks'
            'Armed conflicts and attacks': 'armed conflicts and attacks',
            'Attacks and armed conflicts': 'armed conflicts and attacks',
            'Armed conflicts': 'armed conflicts and attacks',
            'Armed conflict and attacks': 'armed conflicts and attacks',
            'Conflicts and attacks': 'armed conflicts and attacks',
            # 'arts and culture'
            'Arts and culture': 'arts and culture',
            'Art and culture': 'arts and culture',
            'Arts and Culture': 'arts and culture',
            # 'business and economy'
            'Business and economy': 'business and economy',
            'Businesses and economy': 'business and economy',
            'Business and economics': 'business and economy',
            'Business and economic': 'business and economy',
            'Business and Economy': 'business and economy',
            'Business': 'business and economy',
            'Business and the economy': 'business and economy',
            'Economy and finance': 'business and economy',
            # 'disasters and accidents'
            'Disasters and accidents': 'disasters and accidents',
            'Accidents and disasters': 'disasters and accidents',
            'Accidents and Disasters': 'disasters and accidents',
            'Disasters': 'disasters and accidents',
            'Disaster and accidents': 'disasters and accidents',
            # 'law and crime'
            'Law and crime': 'law and crime',
            'Law and Crime': 'law and crime',
            'Law, crime and accidents': 'law and crime',
            # 'politics and elections'
            'Politics and elections': 'politics and elections',
            'Politics': 'politics and elections',
            'Politics and Elections': 'politics and elections',
            # 'science and technology'
            'Science and technology': 'science and technology',
            'Science': 'science and technology',
            'Science and Technology': 'science and technology',
            # 'Sport'
            'Sport': 'sport',
            'Sports': 'sport',
            # 'environment and health', international relations', 'religion'
            'Environment and health': 'environment and health',
            'Health and environment': 'environment and health',
            'Environment': 'environment and health',
            'Health': 'environment and health',
            'Science and environment': 'environment and health',
            'Health and medicine': 'environment and health',
            'Medicine and health': 'environment and health',
            'Pandemics': 'environment and health',
            'International relations': 'international relations',
            'International relation': 'international relations',
            'International Relations': 'international relations',
            'Religion': 'religion'
        }

        classes = {
            0: 'armed conflicts and attacks',
            1: 'arts and culture',
            2: 'business and economy',
            3: 'disasters and accidents',
            4: 'law and crime',
            5: 'politics and elections',
            6: 'science and technology',
            7: 'sport',
            8: 'none'
        }

        article_ids = self.article_dict.keys()
        for article_id in article_ids:
            article = self.article_dict[article_id]

            # remap automatically
            try:
                article.event_type = map_event_type[article.event_type]
            except KeyError:
                article.event_type = article.event_type

            # remap manually
            if article.event_type in ['environment and health', 'religion', 'international relations']:
                print
                print 'Article ID:\t\t' + str(article_id)
                print 'Event type:\t\t' + article.event_type
                print 'Title:\t\t\t' + article.title
                print 'Body:\t\t\t' + ' '.join(article.body.select_subset()[:100])
                event_type = classes[int(raw_input().rstrip())]
                if event_type == 'none':
                    del self.article_dict[article_id]
                else:
                    article.event_type = event_type

        self._save('articles.json', self.article_dict, 'remapped')

    def add_keywords(self, thresh=0.21):
        # put categories in appropriate form (list of strings where each string has ! as delimiter)
        categories = []
        for _, article in self.article_dict.iteritems():
            val = ''
            for i, category in enumerate(article.categories):
                val += category.replace('/', '!').replace('_', ' ').lower()
                if i < len(article.categories)-1:
                    val += '!'
            categories.append(val)

        # build TF-IDF matrix and choose words with highest values as keywords
        vect = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern='[^!]+', lowercase=True, dtype=float)
        cnt = vect.fit_transform(raw_documents=categories)
        trans = TfidfTransformer(norm='l2')
        tfidf = trans.fit_transform(cnt)

        features = vect.get_feature_names()
        for i, (_, article) in enumerate(self.article_dict.iteritems()):
            row = tfidf[i, :].toarray()[0]
            for j in np.where(row > thresh)[0]:
                article.keywords.append(features[j])

        self._save('articles.json', self.article_dict, 'enriched with keywords')

    def add_entities(self):
        for i, (_, article) in enumerate(self.article_dict.iteritems()):
            response = enrych(article.body)
            try:
                root = ET.fromstring(response)
            except ET.ParseError as e:
                print '# %d = %s' % (i, e)
                continue
            for annotation in root.find('annotations').findall('annotation'):
                article.entities.append(annotation.attrib['displayName'])

            # intermediate save
            if i > 0 and i % 50 == 0:
                self._save('articles.json', self.article_dict, 'enriched with entities')

            print '# %d = %d entities' % (i, len(article.entities))

        self._save('articles.json', self.article_dict, 'enriched with entities')

    def preprocess_body(self):
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # remove html markup
            a.body = remove_html_tags(a.body)

        self._save('articles.json', self.article_dict, 'preprocessed (body)')

    def preprocess_metadata(self):
        h = HTMLParser()
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # convert unicode characters to ascii, convert to lower and replace underscores
            a.categories = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.categories]
            a.keywords = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.keywords]
            a.entities = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.entities]

        self._save('articles.json', self.article_dict, 'preprocessed (metadata)')

    def remove_duplicate_metadata(self):
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # uniquify categories, keywords and entities
            a.categories = [val for val in list(set(a.categories)) if (val not in a.keywords) and (val not in a.entities)]
            a.keywords = [val for val in list(set(a.keywords)) if (val not in a.categories) and (val not in a.entities)]
            a.entities = [val for val in list(set(a.entities)) if (val not in a.categories) and (val not in a.keywords)]

        self._save('articles.json', self.article_dict, 'uniquified (metadata)')

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

        self._save('articles_preprocessed.json', self.article_dict, 'cleaned (stemmed, removed stop words)')

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
    def _save(self, file_name, article_dict, desc):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Scraped Wiki article set (%d articles) %s and saved to %s.' % (len(article_dict), desc, path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Scraped Wiki article set (%d articles) loaded from %s.' % (len(articles), path)

        # debug
        # ppr(articles[articles.keys()[0]])

        self.article_dict = {}
        for article_id, a in articles.iteritems():
            article = self._Article()

            article.url = a['url']
            article.title = a['title']
            article.body = a['body']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.entities = a['entities']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict


class WikiJoint:
    #########
    # global
    #########
    def __init__(self, path='data/articles/wiki_guardian/wiki/'):
        self.path = path
        make_dirs(path)
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
            msg += 'Categories:\t%s\n' % str(a.categories)
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
    def join(self, article_dict_list, file_name):
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
    def _save(self, file_name, article_dict):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Joint Wiki article set (%d articles) saved to %s.' % (len(article_dict), path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Joint Wiki article set (%d articles) loaded from %s.' % (len(articles), path)

        # debug
        # ppr(articles[articles.keys()[0]])

        self.article_dict = {}
        for article_id, a in articles.iteritems():
            article = self._Article()

            article.url = a['url']
            article.title = a['title']
            article.body = a['body']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.entities = a['entities']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict


class Guardian:
    #########
    # global
    #########
    def __init__(self, path='data/articles/wiki_guardian/guardian/', event_type=None, n_pages=0):
        self.path = path + event_type + '/'
        make_dirs(self.path)
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
            # ppr(data[0])

            DataSaver.save(self.path + '%d.json' % (page-1), data)

    def parse_add_entities(self):
        if os.path.exists(self.path + 'articles.json'):
            self.article_dict = self.load('articles.json')
        else:
            self.article_dict = {}

        for page in range(self.n_pages):
            articles = DataLoader.load(self.path + '%d.json' % page)

            # debug
            # ppr(articles[0])

            for i, a in enumerate(articles):
                # intermediate save
                if i > 0 and i % 50 == 0:
                    self._save('articles.json', self.article_dict, 'parsed and enriched with entities')

                try:
                    body = a['fields']['body']
                    if a['id'] not in self.article_dict and a['type'] == 'article':
                        article = self._Article()
                        article.url = a['webUrl']
                        article.title = a['fields']['headline']
                        article.body = remove_html_tags(body)
                        response = enrych(article.body)
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
            self._save('articles.json', self.article_dict, 'parsed and enriched with entities')

        print 'Average no. of entities per article:', np.mean([len(a.entities) for _, a in self.article_dict.iteritems()])

        # final save
        self._save('articles.json', self.article_dict, 'parsed and enriched with entities')

    def preprocess_metadata(self):
        h = HTMLParser()
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # convert unicode characters to ascii, convert to lower and replace underscores
            a.categories = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.categories]
            a.keywords = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.keywords]
            a.entities = [unidecode(h.unescape(val.lower().replace('_', ' '))) for val in a.entities]

        self._save('articles.json', self.article_dict, 'preprocessed (metadata)')

    def remove_duplicate_metadata(self):
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # uniquify categories, keywords and entities
            a.categories = [val for val in list(set(a.categories)) if (val not in a.keywords) and (val not in a.entities)]
            a.keywords = [val for val in list(set(a.keywords)) if (val not in a.categories) and (val not in a.entities)]
            a.entities = [val for val in list(set(a.entities)) if (val not in a.categories) and (val not in a.keywords)]

        self._save('articles.json', self.article_dict, 'uniquified (metadata)')

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

        self._save('articles_preprocessed.json', self.article_dict, 'cleaned (stemmed, removed stop words)')

    def count_unique_urls(self):
        urls = []
        for i in range(self.n_pages):
            articles = DataLoader.load(self.path + '%d.json' % i)
            urls += [article['webUrl'] for article in articles]
        print 'Unique URLs in Guardian %s article set: %d / %d' % (self.event_type, len(set(urls)), self.n_pages*200)

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
    def _save(self, file_name, article_dict, desc):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Guardian article set (%d articles) %s and saved to %s.' % (len(article_dict), desc, path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Guardian article set (%d articles) loaded from %s.' % (len(articles), path)

        # debug
        # ppr(articles[articles.keys()[0]])

        self.article_dict = {}
        for article_id, a in articles.iteritems():
            article = self._Article()

            article.url = a['url']
            article.title = a['title']
            article.body = a['body']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.entities = a['entities']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict


class GuardianJoint:
    #########
    # global
    #########
    def __init__(self, path='data/articles/wiki_guardian/guardian/'):
        self.path = path
        make_dirs(path)
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
    def join(self, article_dict_list, file_name):
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
    def _save(self, file_name, article_dict):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Joint Guardian article set (%d articles) saved to %s.' % (len(article_dict), path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Joint Guardian article set (%d articles) loaded from %s.' % (len(articles), path)

        # debug
        # ppr(articles[articles.keys()[0]])

        self.article_dict = {}
        for article_id, a in articles.iteritems():
            article = self._Article()

            article.url = a['url']
            article.title = a['title']
            article.body = a['body']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.entities = a['entities']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict


class WikiGuardian:
    #########
    # global
    #########
    def __init__(self, path='data/articles/wiki_guardian/'):
        self.path = path
        make_dirs(path)
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
            msg += 'Categories:\t%s\n' % str(a.categories)
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
    def join(self, article_dict_list, file_name):
        self.article_dict = article_dict_list[0]
        for i in range(1, len(article_dict_list)):
            for article_id, article in article_dict_list[i].iteritems():
                self.article_dict[article_id] = article

        self._save(file_name, self.article_dict)

    def select_subset(self, article_ids, file_name='', save=False):
        """
        First we select a subset of articles from self.article_dict and store them in article_dict.
        1) If remove trigger is true, we remove selected articles from self.article_dict and save it to file_name.
           We then return the subset of articles stored in article_dict.
        2) If remove trigger is false, we just save the selected subset in article_dict to file_name.
        """
        article_dict = {article_id: self.article_dict[article_id] for article_id in article_ids}
        # if remove:
        #     for article_id in article_ids:
        #         del self.article_dict[article_id]
        #     self._save(file_name, self.article_dict)
        #     return article_dict
        if save:
            self._save(file_name, article_dict)

        return article_dict


    def select_stratified_subset(self, shuffled_article_ids, n_per_event_type=1000):
        selected = defaultdict(list)
        for article_id in shuffled_article_ids:
            article = self.article_dict[article_id]
            if len(selected[article.event_type]) < n_per_event_type:
                selected[article.event_type].append(article_id)

        return selected

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
    def _save(self, file_name, article_dict):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Article set (%d articles) saved to %s.' % (len(article_dict), path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Article set (%d articles) loaded from %s.' % (len(articles), path)

        # debug
        # ppr(articles[articles.keys()[0]])

        self.article_dict = {}
        for article_id, a in articles.iteritems():
            article = self._Article()

            article.url = a['url']
            article.title = a['title']
            article.body = a['body']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.entities = a['entities']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict


class NewsFeed:
    #########
    # global
    #########
    def __init__(self, n, path='data/articles/newsfeed/'):
        self.n = n
        self.path_uncleaned = path + str(n) + '/'
        self.path_cleaned = path + str(n) + '.json'
        self.article_dict = {}

        self.html_entities = ['&quot;', '&amp;', '&lt;', '&gt;', '&circ;', '&tilde;', '&ndash;', '&lsquo;', '&rsquo;',
                              '&sbquo;', '&ldquo;', '&rdquo;', '&bdquo;', '&lsaquo;', '&rsaquo;', '&euro;']
        self.browser_entities = ['"', '&', '<', '>', 'ˆ', '˜', '–', '‘', '’', '‚', '“', '”', '„', '‹', '›', '€']

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
            msg += 'Categories:\t%s\n' % str(a.categories)
            msg += 'Keywords:\t\t%s' % str(a.keywords)
            msg += 'Entities:\t\t\t%s\n' % str(a.entities)
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

    class _Streamer:
        def __init__(self, n, username='zvucko', password='aktivnoucenje', path='data/articles/newsfeed/',
                     feed_url='http://newsfeed.ijs.si/stream/', after=None, n_articles=100):
            self.username = username
            self.password = password
            self.n = n
            self.path = path + str(self.n) + '/'
            make_dirs(self.path)
            self.feed_url = feed_url
            if after is None:
                path = self.path if self.n == 0 else path + str(self.n-1) + '/'
                timestamps = [self._extract_timestamp(fn) for fn in listdir(path) if fn.startswith('public-news')]
                timestamps = [(ts, feed) for (ts, feed) in timestamps if ts]
                feeds = set(feed for (ts, feed) in timestamps)
                feed_latest = [max(ts for (ts, f) in timestamps if f == feed) for feed in feeds]
                self.after = min(feed_latest or ['0000-00-00T00-00-00Z'])
            self.n_articles = n_articles

        @staticmethod
        def _extract_timestamp(file_name):
            """
            If `file_name` contains a ISO 8601 zulu formatted timetamp (yyyy-mm-ddThh:mm:ssZ), return a pair (timestamp,
            file name without timestamp), otherwise, return a pair (None, original file name).
            """
            m = search(r'\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\d(\.\d+)?Z', file_name)
            if m is not None:
                return m.group(0), file_name.replace(m.group(0), '[time]')
            else:
                return None, file_name

        class _Fetcher:
            def __init__(self, feed_urls, start_time, output_dir, prefix_timestamp, no_rych, username, password):
                # Normalize start_time to UTC if there is trailing timezone information
                m = match(r'(.*T.*)([+-])(\d{1,2}):(\d\d)', start_time)  # match the timezone
                if m:
                    d = datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S')
                    delta = -int(m.group(2)+'1') * (int(m.group(3))*3600+int(m.group(4))*60)
                    d += datetime.timedelta(seconds=delta)
                    start_time = d.strftime('%Y-%m-%dT%H:%M:%SZ')
                # Normalize start_time to UTC if a special timezone "L" is given, meaning "local"
                if start_time.endswith('L'):
                    d = datetime.datetime.strptime(start_time[:-1], '%Y-%m-%dT%H:%M:%S')
                    d += datetime.timedelta(seconds=time.timezone)
                    start_time = d.strftime('%Y-%m-%dT%H:%M:%SZ')
                # Server which queries the DB
                self.feed_urls = feed_urls
                # Time of the last fetched article. Continue from here.
                self.last_seen = dict((feed_url, start_time) for feed_url in self.feed_urls)
                # Ids of articles downloaded with the last timestamp (needed to ensure some files with the same
                # timestamp are not skipped)
                self.last_articles_ids = []
                # Directory in which to put the fetched news
                self.output_dir = output_dir
                # Prefix filenames with ISO timestamp when saving to disk?
                self.prefix_timestamp = prefix_timestamp
                # Remove rych fields?
                self.no_rych = no_rych
                # Auth credentials for access to the feed
                self.username = username
                self.password = password

            @staticmethod
            def _extract_timestamp(file_name):
                """
                If `file_name` contains a ISO 8601 zulu formatted timetamp (yyyy-mm-ddThh:mm:ssZ), return a pair (timestamp,
                file name without timestamp), otherwise, return a pair (None, original file name).
                """
                m = search(r'\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\d(\.\d+)?Z', file_name)
                if m is not None:
                    return m.group(0), file_name.replace(m.group(0), '[time]')
                else:
                    return None, file_name

            def run(self, n):
                i = 0
                while i < n:
                    nothing_new = True
                    for feed_url in self.feed_urls:
                        # read data, dump to disk
                        try:
                            url = feed_url + '?after=' + urllib.quote(self.last_seen[feed_url])
                            url += '&last_articles_ids=' + urllib.quote(json.dumps(self.last_articles_ids))
                            if self.no_rych:
                                url += '&norych=on'
                            print 'Trying %r' % url
                            request = urllib2.Request(url)
                            auth = ('%s:%s' % (self.username, self.password)).encode('base64').rstrip('\n')
                            request.add_header('Authorization', 'Basic %s' % auth)
                            q = urllib2.urlopen(request)

                            # we got a file, parse the response
                            nothing_new = False
                            print '  Server has new data, downloading ...'
                            try:
                                filename = '[unknown]'
                                filename = search('filename=[\'"]?([^\'" ;]+)[\'"]?',
                                                  q.headers['content-disposition']).group(1)
                                # extract the timestamp of the freshly fetch file (we'll continue from here)
                                self.last_seen[feed_url], _ = self._extract_timestamp(filename)
                                assert self.last_seen[feed_url] is not None
                            except Exception as exc:
                                raise ValueError(('Server returned a file with a malformed filename: %r; no timestamp '
                                                  'can be parsed. Please report this problem. '
                                                  'Traceback follows.' % filename), exc)
                            try:
                                resp_last_articles_ids = q.info().getheader('X-last-art-ids')
                                self.last_articles_ids = json.loads(resp_last_articles_ids)
                            except Exception as exc:
                                raise ValueError(('Server returned a non-json-parsable list of already '
                                                  'downloaded article ids: %s. Please report this problem. '
                                                  'Traceback follows.' % resp_last_articles_ids), exc)

                            # save the file to disk
                            if self.prefix_timestamp:
                                filename = self._extract_timestamp(filename)[0] + '-' + filename
                            f = open(self.output_dir+'/'+filename, 'wb')
                            f.write(q.read())
                            f.close()
                            print '  Fetched %r (article #%d).' % (filename, i+1)
                            i += 1

                        except Exception as exc:
                            if not (isinstance(exc, urllib2.HTTPError) and exc.getcode() == 404):
                                traceback.print_exc()
                            else:
                                print exc.info()
                        finally:
                            try:
                                q.close()
                            except:
                                pass

                        if nothing_new:
                            print 'No new data yet, sleeping for a minute ...'
                            time.sleep(60)

        def fetch(self):
            fetcher = self._Fetcher(feed_urls=[self.feed_url], start_time=self.after, output_dir=self.path,
                                    prefix_timestamp=False, no_rych=False, username=self.username,
                                    password=self.password)
            print 'Fetching %d articles from newsfeed stream %s and saving them to %s.' % (self.n_articles,
                                                                                           self.feed_url, self.path)
            fetcher.run(self.n_articles)

    #################
    # main functions
    #################
    def fetch_from_newsfeed(self):
        self._Streamer(n=self.n).fetch()

    def parse(self):
        # load uncleaned NewsFeed article sets and parse their XML content
        file_names = [file_name for file_name in listdir(self.path_uncleaned) if file_name.startswith('public-news')]
        print 'Loading and parsing %d NewsFeed article sets from %s.' % (len(file_names), self.path_uncleaned)
        for file_name in file_names:
            root = DataLoader.load(self.path_uncleaned + file_name)

            # debug
            # print root.find('article').find('title').text

            self._parse_single(root)

        self._save('parsed')

    def remove_duplicates(self):
        er = EventRegistry()

        article_ids = self.article_dict.keys()

        n_duplicates = 0

        i = 0
        request_size = 100

        while i < len(article_ids):
            q = QueryArticle(article_ids[i:i+request_size])
            q.addRequestedResult(RequestArticleInfo())
            response = er.execQuery(q)

            for article_id, article in response.iteritems():
                # debug
                # ppr(response[article_id])

                if 'error' not in article:
                    if article['info']['isDuplicate']:
                        del self.article_dict[int(article_id)]
                        n_duplicates += 1

            i += request_size

        print 'Duplicates (%d articles) removed from NewsFeed article set.' % n_duplicates
        self._save('uniquified')

    def summarize(self):
        Summarizer(n=self.n).summarize(self.article_dict)

    def clean(self):
        # initialize Porter stemmer and load stop words
        stemmer = PorterStemmer()
        stop_words = [word.decode('utf-8') for word in DataLoader.load('data/stop_words/stop_words.txt', dtype=str)]

        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # convert characters to lower
            a.title = a.title.lower()
            a.body = a.body.lower()
            a.categories = [val.lower() for val in a.categories]
            a.keywords = [val.lower() for val in a.keywords]
            a.entities = [val.lower() for val in a.entities]
            a.event_type = a.event_type.lower()

            # remove new line characters
            a.title = a.title.replace('\n', ' ')
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

        self._save('cleaned')

    ####################
    # utility functions
    ####################
    def _parse_single(self, root):
        articles = root.findall('article')
        for a in articles:
            if a.find('lang').text == 'eng':
                tags = [element.tag for element in a]

                article = self._Article()

                article.url = self._convert_special_entities(a.find('feed').find('uri').text)

                article.title = a.find('title').text
                if not article.title:
                    article.title = ''

                article.body = ' '.join([val.text for val in a.find('body-cleartext').findall('p')])

                if 'body-rych' in tags:
                    for annotation in a.find('body-rych')[0][2]:
                        if annotation.attrib['type'][9:] in ['location', 'organization', 'person']:
                            article.entities.append(annotation.attrib['displayName'])
                else:
                    for node in a.find('body-xlike').find('item').find('nodes'):
                        if node.attrib['type'] == 'entity' and node.attrib['class'] in ['location', 'organization', 'person']:
                            article.entities.append(node.attrib['displayName'].replace('_', ' '))

                if 'dmoz' in tags:
                    article.categories = [val.text.replace('_', ' ')
                                          for val in a.find('dmoz').find('dmoz-categories').findall('dmoz-category')]

                    article.keywords = [val.text.replace('_', ' ')
                                        for val in a.find('dmoz').find('dmoz-keywords').findall('dmoz-keyword')]
                else:
                    for attribute in a.find('body-rych')[0][0][0]:
                        if attribute.attrib['type'][5:] == 'topic':
                            article.categories.append(attribute.attrib['displayName'].replace('_', ' '))

                        elif attribute.attrib['type'][5:] == 'tag':
                            article.keywords.append(attribute.attrib['displayName'].replace('_', ' '))

                self.article_dict[a.get('id')] = article

    def _convert_special_entities(self, url):
        url = url.encode('utf-8')
        for i in range(len(self.html_entities)):
            url = url.replace(self.html_entities[i], self.browser_entities[i])

        return url

    def _save(self, desc):
        DataSaver.save(self.path_cleaned, {article_id: a.__dict__ for article_id, a in self.article_dict.iteritems()})
        print 'NewsFeed article set (%d articles) %s and saved to %s.' % \
              (len(self.article_dict), desc, self.path_cleaned)

    def load(self):
        articles = DataLoader.load(self.path_cleaned)
        print 'NewsFeed article set (%d articles) loaded from %s.' % (len(articles), self.path_cleaned)

        # debug
        # ppr(articles[articles.keys()[0]])

        self.article_dict = {}
        for article_id, a in articles.iteritems():
            article = self._Article()

            article.title = a['title']
            article.url = a['url']
            article.body = a['body']
            article.categories = a['categories']
            article.keywords = a['keywords']
            article.entities = a['entities']
            article.event_type = a['event_type']

            self.article_dict[article_id] = article

        return self.article_dict


class Features:
    #########
    # global
    #########
    def __init__(self, dir_name='', path='data/features/', labeled=True):
        self.path = path + dir_name + '/'
        make_dirs(self.path)
        self.labeled = labeled

        self.article_ids = []
        self.titles = []
        self.bodies = []
        self.keywords = []
        self.entities = []
        self.nell_categories_nonprob = []
        self.nell_categories_prob = []
        self.event_types = []

    #################
    # main functions
    #################
    def transform(self, article_dict, entity_categories):
        # transform article features to string lists
        for article_id, a in article_dict.iteritems():
            self.article_ids.append(article_id)

            self.titles.append(a.title)

            self.bodies.append(a.body)

            self.keywords.append('!'.join(a.keywords))

            self.entities.append('!'.join(a.entities))

            article_categories_nonprob = defaultdict(int)
            article_categories_prob = defaultdict(float)
            for entity in a.entities + a.keywords:
                for cat in entity_categories[entity]:
                    article_categories_nonprob[cat.name] += 1
                    article_categories_prob[cat.name] += cat.score
            self.nell_categories_nonprob.append('!'.join(sum([[w] * c for w, c in article_categories_nonprob.iteritems()], [])))
            self.nell_categories_prob.append(article_categories_prob)

            if self.labeled:
                self.event_types.append(a.event_type)

        self._save()

    ####################
    # utility functions
    ####################
    def _save(self):
        # save preprocessed features
        DataSaver.save(self.path + 'article_ids.txt', np.array(self.article_ids))
        DataSaver.save(self.path + 'titles.txt', np.array(self.titles))
        DataSaver.save(self.path + 'bodies.txt', np.array(self.bodies))
        DataSaver.save(self.path + 'keywords.txt', np.array(self.keywords))
        DataSaver.save(self.path + 'entities.txt', np.array(self.entities))
        DataSaver.save(self.path + 'nell_categories_nonprob.txt', np.array(self.nell_categories_nonprob))
        DataSaver.save(self.path + 'nell_categories_prob', np.array(self.nell_categories_prob, dtype=object))
        if self.labeled:
            DataSaver.save(self.path + 'event_types.txt', np.array(self.event_types))

        print 'Features transformed and saved to %s.' % self.path

    def load(self):
        # load preprocessed features
        self.article_ids = DataLoader.load(self.path + 'article_ids.txt', skip_empty=False, dtype=unicode)
        self.titles = DataLoader.load(self.path + 'titles.txt', skip_empty=False, dtype=unicode)
        self.bodies = DataLoader.load(self.path + 'bodies.txt', skip_empty=False, dtype=unicode)
        self.keywords = DataLoader.load(self.path + 'keywords.txt', skip_empty=False, dtype=unicode)
        self.entities = DataLoader.load(self.path + 'entities.txt', skip_empty=False, dtype=unicode)
        self.nell_categories_nonprob = DataLoader.load(self.path + 'nell_categories_nonprob.txt', skip_empty=False, dtype=unicode)
        self.nell_categories_prob = DataLoader.load(self.path + 'nell_categories_prob', dtype=object)
        if self.labeled:
            self.event_types = DataLoader.load(self.path + 'event_types.txt', skip_empty=False, dtype=unicode)

        print 'Features loaded from %s.' % self.path
