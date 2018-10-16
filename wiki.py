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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
from EventRegistry import *
from mllib import DataLoader, DataSaver


class WikiOriginal:
    #################
    # global
    #################

    def __init__(self, path='data/articles/wiki/original/'):
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
            # self._ppr(a)

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

    def preprocess_body(self):
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            # remove html markup
            a.body = self._remove_html_tags(a.body)

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
        cat1, key1, ent1 = [], [], []
        cat2, key2, ent2 = [], [], []
        for article_id in self.article_dict:
            a = self.article_dict[article_id]

            cat1.append(len(a.categories))
            key1.append(len(a.keywords))
            ent1.append(len(a.entities))

            # uniquify categories, keywords and entities
            a.categories = [val for val in list(set(a.categories)) if (val not in a.keywords) and (val not in a.entities)]
            a.keywords = [val for val in list(set(a.keywords)) if (val not in a.categories) and (val not in a.entities)]
            a.entities = [val for val in list(set(a.entities)) if (val not in a.categories) and (val not in a.keywords)]

            cat2.append(len(a.categories))
            key2.append(len(a.keywords))
            ent2.append(len(a.entities))

        print np.sum(cat1), np.sum(key1), np.sum(ent1)
        print np.sum(cat2), np.sum(key2), np.sum(ent2)

        self._save('articles2.json', self.article_dict, 'uniquified (metadata)')

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

    @staticmethod
    def _ppr(x):
        pprint.PrettyPrinter(indent=4).pprint(x)

    def _save(self, file_name, article_dict, desc):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Original Wiki article set (%d articles) %s and saved to %s.' % (len(article_dict), desc, path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Original Wiki article set (%d articles) loaded from %s.' % (len(articles), path)

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


class WikiScraped:
    #################
    # global
    #################

    def __init__(self, path='data/articles/wiki/scraped/'):
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
            # self._ppr(response)

            if response:
                for article_id in response:
                    # debug
                    # self._ppr(response[article_id])

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
            response = self._enrych(article.body)
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
            a.body = self._remove_html_tags(a.body)

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

    @staticmethod
    def _ppr(x):
        pprint.PrettyPrinter(indent=4).pprint(x)

    def _save(self, file_name, article_dict, desc):
        path = self.path + file_name
        DataSaver.save(path, {article_id: a.__dict__ for article_id, a in article_dict.iteritems()})
        print 'Scraped Wiki article set (%d articles) %s and saved to %s.' % (len(article_dict), desc, path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Scraped Wiki article set (%d articles) loaded from %s.' % (len(articles), path)

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


class Wiki:
    #################
    # global
    #################

    def __init__(self, path='data/articles/wiki/'):
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
        print 'Joint Wiki article set (%d articles) saved to %s.' % (len(article_dict), path)

    def load(self, file_name):
        path = self.path + file_name
        articles = DataLoader.load(path)
        print 'Joint Wiki article set (%d articles) loaded from %s.' % (len(articles), path)

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
