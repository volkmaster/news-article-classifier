#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'


from os import makedirs
import errno
from collections import defaultdict, OrderedDict
from itertools import islice
import pprint
import numpy as np
from scipy.sparse import hstack, lil_matrix
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import requests
from EventRegistry import *
from mllib import DataLoader, DataSaver, DataPreprocessor, CrossValidation, Evaluator


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


#######
# dump
#######
class Logger:
    def __init__(self, type_, path='data/log/'):
        # type_:
        #   - 'bp': batch preprocessor
        #   - 'nsp': news stream preprocessor,
        #   - 'al': active learning,
        #   - 'cv': cross-validation,
        #   - 'test': debug
        self.path = path + 'log_' + type_ + '.txt'
        make_dirs(path)

    def __call__(self, msg):
        fp = open(self.path, 'a')
        fp.write(str(datetime.datetime.now()) + ': ' + msg + '\n')
        fp.close()


class Config:
    def __init__(self, logger, path_config='data/config/'):
        self.logger = logger

        # batch preprocessor (bp)
        self.iter_bp = -1
        self.running_bp = False
        self.path_bp = path_config + 'bp.json'

        # news stream preprocessor (nsp)
        self.iter_nsp = -1
        self.running_nsp = False
        self.path_nsp = path_config + 'nsp.json'

        # active learning (al)
        self.iter_al = -1
        self.paused_al = False
        self.path_al = path_config + 'al.json'

        # cross-validation (cv)
        self.c = []
        self.idx = -1
        self.iter_cv = -1
        self.running_cv = False
        self.path_cv = path_config + 'cv.json'

    def load(self):
        data = DataLoader.load(self.path_bp)
        self.iter_bp = data['iter']
        self.running_bp = data['running']

        data = DataLoader.load(self.path_nsp)
        self.iter_nsp = data['iter']
        self.running_nsp = data['running']

        data = DataLoader.load(self.path_al)
        self.iter_al = data['iter']
        self.paused_al = data['paused']

        data = DataLoader.load(self.path_cv)
        self.c = data['c']      # SVC parameter
        self.idx = data['idx']  # matrix type
        self.iter_cv = data['iter']
        self.running_cv = data['running']

    def stop(self, type_):
        if type_ == 'bp':
            # indicate that the batch preprocessor has stopped / not running anymore
            self.running_bp = False
            DataSaver.save(self.path_bp, {'iter': self.iter_bp, 'running': self.running_bp})
            self.logger('Running trigger dropped.')
        elif type_ == 'nsp':
            # indicate that the news stream preprocessor has stopped / not running anymore
            self.running_nsp = False
            DataSaver.save(self.path_nsp, {'iter': self.iter_nsp, 'running': self.running_nsp})
            self.logger('Running trigger dropped.')
        elif type_ == 'cv':
            # indicate that the cross-validator has stopped / not running anymore
            self.running_cv = False
            DataSaver.save(self.path_cv, {'c': self.c, 'idx': self.idx, 'iter': self.iter_cv, 'running': self.running_cv})
            self.logger('Running trigger dropped.')

    def unpause(self):
        # unpause the active learning loop
        self.paused_al = False
        DataSaver.save(self.path_al, {'iter': self.iter_al, 'paused': self.paused_al})
        self.logger('Pause trigger dropped.')

    def increment_iter(self, type_):
        if type_ == 'nsp':
            # increment the news stream preprocessor iteration no.
            self.iter_nsp += 1
            DataSaver.save(self.path_nsp, {'iter': self.iter_nsp, 'running': self.running_nsp})
            self.logger('News stream preprocessor iteration no. incremented to %d.' % self.iter_nsp)
        elif type_ == 'al':
            # increment the active learning iteration no.
            self.iter_al += 1
            DataSaver.save(self.path_al, {'iter': self.iter_al, 'paused': self.paused_al})
            self.logger('Active learning iteration no. incremented to %d.' % self.iter_al)
        elif type_ == 'cv':
            # increment the cross-validation iteration no.
            self.iter_cv += 1
            DataSaver.save(self.path_cv, {'c': self.c, 'idx': self.idx, 'iter': self.iter_cv, 'running': self.running_cv})
            self.logger('Cross-validation iteration no. incremented to %d.' % self.iter_cv)


class Labeled:
    def __init__(self, logger, n, path='data/articles/batch/', path_predictions='data/predictions/'):
        self.logger = logger

        self.n = n
        self.path = path
        self.path_cleaned = path + str(n) + '.json'
        self.path_evaluated_predictions = path_predictions + 'evaluated/' + str(n) + '.json'
        self.article_dict = {}

    def create_new(self, article_dict1, article_dict2):
        # load evaluated predictions evaluated with active learning
        evaluated_predictions = DataLoader.load(self.path_evaluated_predictions)
        self.logger('Evaluated predictions (%d articles) loaded from %s.' % (len(evaluated_predictions),
                                                                             self.path_evaluated_predictions))

        self.article_dict = article_dict1
        for article_id, label in evaluated_predictions.iteritems():
            article_id = int(article_id)
            if label != 'none':
                self.article_dict[article_id] = article_dict2[article_id]
                self.article_dict[article_id].event_type = label

        # save concatenated batch article set
        # self._save('concatenated')


###############
# main classes
###############
class Summarizer:
    #########
    # global
    #########
    def __init__(self, n, path='data/articles/newsfeed/summaries/'):
        self.path = path + str(n) + '.json'
        make_dirs(path)
        self.summary_dict = {}

    def __contains__(self, article_id):
        return article_id in self.summary_dict

    def __call__(self, article_id):
        msg = ''
        if article_id in self:
            s = self.summary_dict[article_id]
            msg += 'Article ID:\t\t\t%s\n' % article_id
            msg += 'Title:\t\t\t\t%s\n' % s.title
            msg += 'Body:\t\t\t\t%s' % s.body
        else:
            msg = 'No summary with such ID.'

        return msg

    class _Summary:
        def __init__(self):
            self.title = ''
            self.body = ''

    #################
    # main functions
    #################
    def summarize(self, article_dict):
        for article_id, a in article_dict.iteritems():
            summary = self._Summary()
            summary.title = a.title
            summary.body = ' '.join(a.body.select_subset()[:120])
            self.summary_dict[article_id] = summary

        # save article summaries used for user evaluation part of active learning
        self._save()

    ####################
    # utility functions
    ####################
    def _save(self):
        DataSaver.save(self.path, {article_id: s.__dict__ for article_id, s in self.summary_dict.iteritems()})
        print 'Summaries (%d articles) saved to %s.' % (len(self.summary_dict), self.path)

    def load(self):
        summaries = DataLoader.load(self.path)
        print 'Summaries (%d articles) loaded from %s.' % (len(summaries), self.path)

        # debug
        # ppr(summaries[summaries.keys()[0]])

        for article_id, s in summaries.iteritems():
            summary = self._Summary()

            summary.title = s['title']
            summary.body = s['body']

            self.summary_dict[article_id] = summary

        return self.summary_dict


class NELLDict:
    #########
    # global
    #########
    def __init__(self, path='data/nell/entity_categories.json', trace=False):
        self.path = path
        self.trace = trace
        self.NELL_API_URL = 'http://rtw.ml.cmu.edu/rtw/api/json0'
        self.QUERY_SUCCESS = 'NELLQueryDemoJSON0'
        self.entity_categories = defaultdict(list)

    class _Category:
        def __init__(self):
            self.name = ''
            self.score = 0.0

    #################
    # main functions
    #################
    def update(self, article_dict):
        # query NELL API and add new entity categories to dict
        for i, (article_id, article) in enumerate(article_dict.iteritems()):
            if i > 0 and i % 50 == 0:
                # intermediate save
                self._save()
                if self.trace:
                    print '***(%d/%d): %d NELL entity categories saved to %s.' % (i, len(article_dict),
                                                                                  len(self.entity_categories),
                                                                                  self.path)

            # debug
            # print i

            for entity in article.entities + article.keywords:
                if entity not in self.entity_categories:
                    self._query(entity)

        # final save
        self._save()
        print '%d NELL entity categories saved to %s.' % (len(self.entity_categories), self.path)

    ####################
    # utility functions
    ####################
    def _query(self, entity):
        # HTTP request & response
        try:
            params = dict(lit1=entity.replace(' ', '_'), predicate='*')
            response = requests.get(self.NELL_API_URL, params)
            data = json.loads(response.text)
        except requests.exceptions.ConnectionError:
            return

        # debug
        # ppr(data)

        # success
        if data['kind'] == self.QUERY_SUCCESS:

            # no categories returned
            if not data['items']:
                return

            # categories found, parse the response and store it
            else:
                names = []
                for item in data['items']:
                    name = item['predicate'].encode('utf-8')
                    if name not in names:
                        cat = self._Category()
                        cat.name = name
                        cat.score = float(item['justifications'][0]['score'])
                        self.entity_categories[entity].append(cat)
                        names.append(name)

                if self.trace:
                    msg = entity + ': '
                    for cat in self.entity_categories[entity]:
                        msg += '(' + cat.name + ', ' + str(cat.score) + ') '
                    print msg
        # error
        else:
            # debug
            # print data['message']
            return

    def _save(self):
        DataSaver.save(self.path, {e: [c.__dict__ for c in cat] for e, cat in self.entity_categories.iteritems()})

    def load(self):
        # load existing NELL entity categories
        entity_categories_dict = DataLoader.load(self.path)
        print '%d NELL entity categories loaded from %s.' % (len(entity_categories_dict), self.path)

        # debug
        # ppr(entity_categories_dict['soviet union'])

        for entity, categories in entity_categories_dict.iteritems():
            for category in categories:
                cat = self._Category()
                cat.name = category['name']
                cat.score = category['score']
                self.entity_categories[entity].append(cat)


class TFIDF:
    #########
    # global
    #########
    def __init__(self, idx=0, norm='l2'):
        self.idx = idx
        self.norm = norm
        self.train = None
        self.test = None

    #################
    # main functions
    #################
    def construct(self, features_train, features_test=None):
        # build sparse document-term matrix using count vectorizer and apply TF-IDF normalization using transformer
        # prepare count feature matrices
        # unigrams
        vect_titles = CountVectorizer(analyzer='word', ngram_range=(1, 1),
                                      lowercase=False, dtype=float)
        cnt_titles_train = vect_titles.fit_transform(features_train.titles)
        if features_test:
            cnt_titles_test = vect_titles.transform(features_test.titles)

        vect_bodies = CountVectorizer(analyzer='word', ngram_range=(1, 1),
                                      lowercase=False, dtype=float)
        cnt_bodies_train = vect_bodies.fit_transform(features_train.bodies)
        if features_test:
            cnt_bodies_test = vect_bodies.transform(features_test.bodies)

        # entities & keywords
        if self.idx in [1, 4, 5]:
            vect_entities = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern='[^!]+',
                                            lowercase=False, dtype=float)
            cnt_entities_train = vect_entities.fit_transform(raw_documents=features_train.entities)
            if features_test:
                cnt_entities_test = vect_entities.transform(features_test.entities)

            vect_keywords = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern='[^!]+',
                                            lowercase=False, dtype=float)
            cnt_keywords_train = vect_keywords.fit_transform(raw_documents=features_train.keywords)
            if features_test:
                cnt_keywords_test = vect_keywords.transform(features_test.keywords)

        # nell_categories_nonprob
        if self.idx in [2, 4]:
            vect_nell_categories = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern='[^!]+',
                                                   lowercase=False, dtype=float)
            cnt_nell_categories_nonprob_train = vect_nell_categories.fit_transform(raw_documents=features_train.nell_categories_nonprob)
            if features_test:
                cnt_nell_categories_nonprob_test = vect_nell_categories.transform(features_test.nell_categories_nonprob)

        # nell_categories_prob
        if self.idx in [3, 5]:
            vect_nell_categories = CountVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern='[^!]+',
                                                   lowercase=False, dtype=float)
            cnt_nell_categories_nonprob_train = vect_nell_categories.fit_transform(raw_documents=features_train.nell_categories_nonprob)
            features = vect_nell_categories.get_feature_names()
            cnt_nell_categories_prob_train = self._prob(cnt_nell_categories_nonprob_train, features_train.nell_categories_prob, features)
            if features_test:
                cnt_nell_categories_nonprob_test = vect_nell_categories.transform(features_test.nell_categories_nonprob)
                cnt_nell_categories_prob_test = self._prob(cnt_nell_categories_nonprob_test, features_test.nell_categories_prob, features)

        # join features into single matrix and apply TF-IDF normalization
        # 0: 'unigrams'
        if self.idx == 0:
            trans = TfidfTransformer(norm=self.norm)
            self.train = trans.fit_transform(hstack((cnt_titles_train, cnt_bodies_train)))

            if features_test:
                self.test = trans.transform(hstack((cnt_titles_test, cnt_bodies_test)))

        # 1: 'unigrams + entities + keywords'
        elif self.idx == 1:
            trans1 = TfidfTransformer(norm=self.norm)
            train1 = trans1.fit_transform(hstack((cnt_titles_train, cnt_bodies_train)))
            trans2 = TfidfTransformer(norm=self.norm)
            train2 = trans2.fit_transform(hstack((cnt_entities_train, cnt_keywords_train)))
            self.train = hstack((train1, train2))

            if features_test:
                test1 = trans1.transform(hstack((cnt_titles_test, cnt_bodies_test)))
                test2 = trans2.transform(hstack((cnt_entities_test, cnt_keywords_test)))
                self.test = hstack((test1, test2))

        # 2: 'unigrams + nell_categories_nonprob'
        elif self.idx == 2:
            trans = TfidfTransformer(norm=self.norm)
            self.train = trans.fit_transform(hstack((cnt_titles_train, cnt_bodies_train, cnt_nell_categories_nonprob_train)))

            if features_test:
                self.test = trans.transform(hstack((cnt_titles_test, cnt_bodies_test, cnt_nell_categories_nonprob_test)))

        # 3: 'unigrams + nell_categories_prob'
        elif self.idx == 3:
            trans = TfidfTransformer(norm=self.norm)
            self.train = trans.fit_transform(hstack((cnt_titles_train, cnt_bodies_train, cnt_nell_categories_prob_train)))

            if features_test:
                self.test = trans.transform(hstack((cnt_titles_test, cnt_bodies_test, cnt_nell_categories_prob_test)))

        # 4: 'unigrams + entities + keywords + nell_categories_nonprob'
        elif self.idx == 4:
            trans1 = TfidfTransformer(norm=self.norm)
            train1 = trans1.fit_transform(hstack((cnt_titles_train, cnt_bodies_train, cnt_nell_categories_nonprob_train)))
            trans2 = TfidfTransformer(norm=self.norm)
            train2 = trans2.fit_transform(hstack((cnt_entities_train, cnt_keywords_train)))
            self.train = hstack((train1, train2))

            if features_test:
                test1 = trans1.transform(hstack((cnt_titles_test, cnt_bodies_test, cnt_nell_categories_nonprob_test)))
                test2 = trans2.transform(hstack((cnt_entities_test, cnt_keywords_test)))
                self.test = hstack((test1, test2))

        # 5: 'unigrams + entities + keywords + nell_categories_prob'
        elif self.idx == 5:
            trans1 = TfidfTransformer(norm=self.norm)
            train1 = trans1.fit_transform(hstack((cnt_titles_train, cnt_bodies_train, cnt_nell_categories_prob_train)))
            trans2 = TfidfTransformer(norm=self.norm)
            train2 = trans2.fit_transform(hstack((cnt_entities_train, cnt_keywords_train)))
            self.train = hstack((train1, train2))

            if features_test:
                test1 = trans1.transform(hstack((cnt_titles_test, cnt_bodies_test, cnt_nell_categories_prob_test)))
                test2 = trans2.transform(hstack((cnt_entities_test, cnt_keywords_test)))
                self.test = hstack((test1, test2))

        if features_test:
            return self.train, self.test
        else:
            return self.train

    ####################
    # utility functions
    ####################
    @staticmethod
    def _prob(mat, nell_categories_prob, features):
        new_mat = lil_matrix(mat.shape, dtype=np.float64)
        features = {v: k for k, v in dict(enumerate(features)).iteritems()}
        for i in range(new_mat.shape[0]):
            for name, score in nell_categories_prob[i].iteritems():
                try:
                    new_mat[i, features[name]] = score
                except KeyError:
                    pass

        return new_mat.tocsr()


class Classifier:
    #########
    # global
    #########
    def __init__(self, path_results='data/results/',
                 mode='cv',
                 k=5,
                 approach='random',
                 n=0,
                 skewed_class_weighing=True,
                 feature_indices=[0],
                 combinations={'svm': [1]},
                 shuffle=False,
                 seed=-1,
                 trace=False):
        # general
        # modes: 1) 'cv': cross-validation to evaluate models built with different combinations of parameters,
        #        2) 'model_evaluation': evaluate model on the test set with optimal combination of parameters,
        #        3) 'active_learning': predict labels and probabilities on active learning set / newsfeed article set
        #                              for uncertainty-based instance selection
        self.mode = mode
        self.approach = approach                            # 'init'/'random'/'margin_approach'/'correlation'/'utility'
        self.feature_indices = feature_indices              # determines the selection of features in TF-IDF matrix
        self.combinations = combinations                    # dict with algorithms as keys and params as values
        self.skewed_class_weighing = skewed_class_weighing  # weighing for unbalanced class distribution
        self.shuffle = shuffle                              # shuffle the examples of dataset before learning
        self.seed = seed
        self.trace = trace

        if mode == 'cv':
            self.k = k
            self.path_confusion_matrix = path_results + mode + '/confusion_matrix/'
            self.path_metrics = path_results + mode + '/metrics/'
            make_dirs(self.path_confusion_matrix)
            make_dirs(self.path_metrics)
        elif mode in ['model_evaluation', 'active_learning']:
            self.n = n
            self.path_confusion_matrix = path_results + mode + '/' + approach + '/confusion_matrix/' + str(n) + '_'
            self.path_metrics = path_results + mode + '/' + approach + '/metrics/' + str(n) + '_'
            make_dirs(path_results + mode + '/' + approach + '/confusion_matrix/')
            make_dirs(path_results + mode + '/' + approach + '/metrics/')

        # attributes
        self.train = None
        self.test = None
        self.feature_spaces = {0: 'unigrams',
                               1: 'unigrams + entities + keywords',
                               2: 'unigrams + nell_categories_nonprob',
                               3: 'unigrams + nell_categories_prob',
                               4: 'unigrams + entities + keywords + nell_categories_nonprob',
                               5: 'unigrams + entities + keywords + nell_categories_prob'}
        self.classes = []
        self.results = {}

    #################
    # main functions
    #################
    def fit_predict(self, train, test=None):
        self.train = train
        self.test = test

        # initialize results dict
        self._init_results_dict()

        # shuffle the examples in the training set (optional)
        self._shuffle()

        ################################################################################################################
        # CROSS-VALIDATION: build model instances, predict and evaluate results with k-fold cross-validation
        ################################################################################################################
        if self.mode == 'cv':
            if self.trace:
                print
                print 130 * '-'
                print 'Cross-validation (k=%d):' % self.k

            # encode class labels
            y, classes = DataPreprocessor.encode_labels(self.train.event_types)
            self.classes = classes

            i = 1
            for idx_train, idx_test in CrossValidation.divide_range(len(y), k=self.k):
                # build i-th model instance
                train, test = self._select_examples_cv(idx_train, idx_test)

                for idx in self.feature_indices:
                    # build train and test TF-IDF matrices
                    x_train, x_test = TFIDF(idx).construct(train, test)
                    y_train, y_test = y[idx_train], y[idx_test]

                    # determine class weights for better hyperplane separation of skewed classes (optional)
                    class_weights = self._calculate_class_weights(y_train)

                    for algorithm in self.combinations:
                        for param in self.combinations[algorithm]:
                            # build classifier on the training set for every combination of algorithms and parameters
                            if self.trace:
                                print '# %d *** classifying (features: %s [# %d], algorithm: %s, param: %s)...' % \
                                      (i, self.feature_spaces[idx], x_train.shape[1], algorithm, param)

                            clf = None

                            if algorithm == 'svm':
                                clf = SVC(C=param, kernel='linear', class_weight=class_weights).\
                                    fit(X=x_train, y=y_train)

                            elif algorithm == 'logreg':
                                clf = LogisticRegression(C=param, class_weight=class_weights).\
                                    fit(X=x_train, y=y_train)

                            elif algorithm == 'rf':
                                clf = RandomForestClassifier(n_estimators=param, class_weight=class_weights, n_jobs=-1).\
                                    fit(X=x_train, y=y_train)

                            elif algorithm == 'knn':
                                clf = KNeighborsClassifier(n_neighbors=param, weights='distance').\
                                    fit(X=x_train, y=y_train)

                            # predict on the test set (classes) and store results
                            y_true, y_pred = y_test, clf.predict(x_test)
                            self._store_result(idx, algorithm, param, y_true, y_pred)

                i += 1

        ################################################################################################################
        # MODEL EVALUATION: build model on the training set and predict on the test set
        ################################################################################################################
        elif self.mode == 'model_evaluation':
            if self.trace:
                print
                print 130 * '-'
                print 'Model evaluation (n=%d):' % self.n

            # encode class labels
            y_train, classes = DataPreprocessor.encode_labels(self.train.event_types)
            y_test, _ = DataPreprocessor.encode_labels(self.test.event_types)
            self.classes = classes

            idx = self.feature_indices[0]
            algorithm = self.combinations.keys()[0]
            param = self.combinations[algorithm][0]

            # construct training and test set TF-IDF matrices
            x_train, x_test = TFIDF(idx).construct(self.train, self.test)

            # determine class weights for better hyperplane separation of skewed classes (optional)
            class_weights = self._calculate_class_weights(y_train)

            # build classifier on the training set
            if self.trace:
                print 'classifying (features: %s [# %d], algorithm: %s, param: %s)...' % \
                      (self.feature_spaces[idx], x_train.shape[1], algorithm, param)

            clf = None

            if algorithm == 'svm':
                clf = SVC(C=param, kernel='linear', class_weight=class_weights).\
                    fit(X=x_train, y=y_train)

            elif algorithm == 'logreg':
                clf = LogisticRegression(C=param, class_weight=class_weights).\
                    fit(X=x_train, y=y_train)

            elif algorithm == 'rf':
                clf = RandomForestClassifier(n_estimators=param, class_weight=class_weights, n_jobs=-1).\
                    fit(X=x_train, y=y_train)

            elif algorithm == 'knn':
                clf = KNeighborsClassifier(n_neighbors=param, weights='distance').\
                    fit(X=x_train, y=y_train)

            # predict on the test set (classes and probabilities) and store results
            y_true, y_pred = y_test, clf.predict(x_test)
            self._store_result(idx, algorithm, param, y_true, y_pred)

        ################################################################################################################
        # ACTIVE LEARNING: build model on the training set and predict on the test set (with probabilities)
        ################################################################################################################
        elif self.mode == 'active_learning':
            if self.trace:
                print
                print 130 * '-'
                print 'Active learning uncertainty-based instance selection (n=%d):' % self.n

            # encode class labels
            y_train, classes = DataPreprocessor.encode_labels(self.train.event_types)
            y_test, _ = DataPreprocessor.encode_labels(self.test.event_types)
            self.classes = classes

            idx = self.feature_indices[0]
            algorithm = self.combinations.keys()[0]
            param = self.combinations[algorithm][0]

            # construct training and test set TF-IDF matrices
            x_train, x_test = TFIDF(self.feature_indices[0]).construct(self.train, self.test)

            # determine class weights for better hyperplane separation of skewed classes (optional)
            class_weights = self._calculate_class_weights(y_train)

            # build classifier on the training set
            if self.trace:
                print 'Classifying (Features: %s [# %d], Algorithm: %s, Param: %s)...' % \
                      (self.feature_spaces[idx], x_train.shape[1], algorithm, param)

            clf = None

            if algorithm == 'svm':
                clf = SVC(C=param, kernel='linear', class_weight=class_weights, probability=True, random_state=self.seed).\
                    fit(X=x_train, y=y_train)

            elif algorithm == 'logreg':
                clf = LogisticRegression(C=param, class_weight=class_weights, random_state=self.seed).\
                    fit(X=x_train, y=y_train)

            elif algorithm == 'rf':
                clf = RandomForestClassifier(n_estimators=param, class_weight=class_weights, random_state=self.seed, n_jobs=-1).\
                    fit(X=x_train, y=y_train)

            elif algorithm == 'knn':
                clf = KNeighborsClassifier(n_neighbors=param, weights='distance').\
                    fit(X=x_train, y=y_train)

            # predict on the test set (classes and probabilities) and store results
            y_true, y_pred, y_prob = y_test, clf.predict(x_test), clf.predict_proba(x_test)
            self._store_result(idx, algorithm, param, y_true, y_pred, y_prob)

        return self.results

    def evaluate_and_save_results(self):
        params_best = (0, 'svm', 1)
        fscore_best = -1.0
        top = ['Features', 'Algorithm', 'Parameter', 'Classification accuracy', 'Precision', 'Recall', 'F1-score']
        left = []
        mat_results = np.zeros(shape=(len(self.feature_indices) * len(self.combinations) *
                                      len(self.combinations[self.combinations.keys()[0]]), 4),
                               dtype=float)

        print
        print 130 * '-'
        print 'Results:'

        i = 0
        for idx in self.results:
            for algorithm in self.results[idx]:
                for param in self.results[idx][algorithm]:
                    y_true = self.results[idx][algorithm][param]['true']
                    y_pred = self.results[idx][algorithm][param]['pred']

                    if self.mode == 'cv':
                        fscore = Evaluator.fscore(y_true, y_pred)
                        if fscore > fscore_best:
                            fscore_best = fscore
                            params_best = (idx, algorithm, param)

                    # print results
                    print 'Model # %d (features: %s, algorithm: %s, param: %s)' % (i+1, self.feature_spaces[idx],
                                                                                   algorithm, param)
                    print '\tCA = %.4f (%d/%d)' % Evaluator.ca(y_true, y_pred)
                    print '\tPrecision = %.4f' % Evaluator.precision(y_true, y_pred)
                    print '\tRecall = %.4f' % Evaluator.recall(y_true, y_pred)
                    print '\tF1-score = %.4f' % Evaluator.fscore(y_true, y_pred)
                    print

                    # save confusion matrix and metrics for individual models separately
                    path_confusion_matrix = self.path_confusion_matrix + 'idx=%d,algorithm=%s,param=%d.csv' % \
                                                                         (idx, algorithm, param)
                    path_metrics = self.path_metrics + 'idx=%d,algorithm=%s,param=%d.csv' % (idx, algorithm, param)
                    Evaluator.classification_results_to_csv(y_true, y_pred,
                                                            classes=self.classes,
                                                            path_confusion_matrix=path_confusion_matrix,
                                                            path_metrics=path_metrics)

                    if self.mode == 'cv':
                        left.append('%s;%s;%d' % (self.feature_spaces[idx], algorithm, param))
                        mat_results[i, 0] = Evaluator.ca(y_true, y_pred)[0]
                        mat_results[i, 1] = Evaluator.precision(y_true, y_pred)
                        mat_results[i, 2] = Evaluator.recall(y_true, y_pred)
                        mat_results[i, 3] = Evaluator.fscore(y_true, y_pred)

                    i += 1

        if self.mode == 'cv':
            # save metrics for all models jointly
            DataSaver.save(self.path_metrics + 'cv.csv', mat_results, top=top, left=left)

            # print best model parameters
            idx, algorithm, param = params_best
            y_true = self.results[idx][algorithm][param]['true']
            y_pred = self.results[idx][algorithm][param]['pred']

            print 130 * '-'
            print 'Best model (features: %s, algorithm: %s, param: %s)' % (self.feature_spaces[idx], algorithm, param)
            print '\tCA = %.4f (%d/%d)' % Evaluator.ca(y_true, y_pred)
            print '\tPrecision = %.4f' % Evaluator.precision(y_true, y_pred)
            print '\tRecall = %.4f' % Evaluator.recall(y_true, y_pred)
            print '\tF1-score = %.4f' % Evaluator.fscore(y_true, y_pred)

    ####################
    # utility functions
    ####################
    def _shuffle(self):
        if self.shuffle:
            if self.seed >= 0:
                np.random.seed(self.seed)
            order = np.arange(len(self.train.article_ids))
            np.random.shuffle(order)

            self.train.article_ids = self.train.article_ids[order]
            self.train.titles = self.train.titles[order]
            self.train.bodies = self.train.bodies[order]
            self.train.keywords = self.train.keywords[order]
            self.train.entities = self.train.entities[order]
            self.train.nell_categories_nonprob = self.train.nell_categories_nonprob[order]
            self.train.nell_categories_prob = self.train.nell_categories_prob[order]
            self.train.event_types = self.train.event_types[order]

    def _select_examples_cv(self, idx_train, idx_test):
        class _Features:
            def __init__(self):
                self.titles = []
                self.bodies = []
                self.keywords = []
                self.entities = []
                self.nell_categories_nonprob = []
                self.nell_categories_prob = []

        train = _Features()
        train.titles = self.train.titles[idx_train]
        train.bodies = self.train.bodies[idx_train]
        train.keywords = self.train.keywords[idx_train]
        train.entities = self.train.entities[idx_train]
        train.nell_categories_nonprob = self.train.nell_categories_nonprob[idx_train]
        train.nell_categories_prob = self.train.nell_categories_prob[idx_train]

        test = _Features()
        test.titles = self.train.titles[idx_test]
        test.bodies = self.train.bodies[idx_test]
        test.keywords = self.train.keywords[idx_test]
        test.entities = self.train.entities[idx_test]
        test.nell_categories_nonprob = self.train.nell_categories_nonprob[idx_test]
        test.nell_categories_prob = self.train.nell_categories_prob[idx_test]

        return train, test

    def _calculate_class_weights(self, y_train):
        if self.skewed_class_weighing:
            weights = {}
            for cl in self.classes:
                pos = len(y_train[y_train == cl])
                weights[cl] = (len(y_train) - pos) / float(pos)
        else:
            weights = {cl: 1 for cl in self.classes}

        return weights

    def _init_results_dict(self):
        for idx in self.feature_indices:
            self.results[idx] = {}
            for algorithm in self.combinations:
                self.results[idx][algorithm] = {}
                for param in self.combinations[algorithm]:
                    self.results[idx][algorithm][param] = {}
                    self.results[idx][algorithm][param]['true'] = np.array([])
                    self.results[idx][algorithm][param]['pred'] = np.array([])

    def _store_result(self, idx, algorithm, param, y_true, y_pred, y_prob=None):
        if self.mode == 'cv':
            self.results[idx][algorithm][param]['true'] = np.concatenate((self.results[idx][algorithm][param]['true'], y_true))
            self.results[idx][algorithm][param]['pred'] = np.concatenate((self.results[idx][algorithm][param]['pred'], y_pred))

        elif self.mode == 'model_evaluation':
            self.results[idx][algorithm][param]['true'] = y_true
            self.results[idx][algorithm][param]['pred'] = y_pred

        elif self.mode == 'active_learning':
            self.results[idx][algorithm][param]['true'] = y_true
            self.results[idx][algorithm][param]['pred'] = y_pred
            self.results[idx][algorithm][param]['prob'] = y_prob


class ActiveLearning:
    #########
    # global
    #########
    def __init__(self, n, approach, n_subset, features_train, features_active_learning,
                 feature_indices=[5], combinations={'svm': [100]}, seed=-1, beta=2):
        self.n = n
        self.approach = approach        # 'random'/'margin'/'entropy'/'margin-correlation'/'entropy-correlation'
        self.n_subset = n_subset
        self.seed = seed

        # uncertainty-based instance selection
        self.features_train = features_train
        self.features_active_learning = features_active_learning
        self.feature_indices = feature_indices
        self.combinations = combinations

        # correlation-based instance selection
        self.beta = beta

    #################
    # main functions
    #################
    def select(self):
        article_ids = np.array(self.features_active_learning.article_ids)

        ##################################
        # random-based instance selection
        ##################################
        if self.approach == 'random':
            # choose a random subset of articles from active learning set
            order = np.arange(len(self.features_active_learning.article_ids))
            np.random.seed(self.seed)
            np.random.shuffle(order)
            article_ids = self.features_active_learning.article_ids[order][:self.n_subset]

        ##################################
        # margin-based instance selection
        ##################################
        elif self.approach in ['margin']:
            # run classification to retrieve instance prediction probabilities
            y_prob = self._classify_and_get_probabilities()

            # calculate margin between posterior probabilities of two most likely class labels
            margins = self._calculate_margins(y_prob, article_ids)

            # sort in ascending order and choose first n_subset article ids
            article_ids = list(islice(OrderedDict(sorted(margins.iteritems(), key=lambda x: x[1], reverse=False)),
                                      self.n_subset))

        ###################################
        # entropy-based instance selection
        ###################################
        elif self.approach in ['entropy']:
            # run classification to retrieve instance prediction probabilities
            y_prob = self._classify_and_get_probabilities()

            # calculate uncertainty over the whole output prediction distribution (entropy)
            entropies = self._calculate_entropies(y_prob, article_ids)

            # sort in descending order and choose first n_subset article ids
            article_ids = list(islice(OrderedDict(sorted(entropies.iteritems(), key=lambda x: x[1], reverse=True)),
                                      self.n_subset))

        ##############################################
        # margin-correlation-based instance selection
        ##############################################
        elif self.approach in ['margin-correlation']:
            # run classification to retrieve instance prediction probabilities
            y_prob = self._classify_and_get_probabilities()

            # calculate margin between posterior probabilities of two most likely class labels
            margins = self._calculate_margins(y_prob, article_ids)

            # find most centered instances based on cosine similarity
            correlations = self._calculate_correlations(article_ids)

            # calculate utility values (we subtract margins from 1, since we are looking for examples with maximum
            # utility values)
            utilities = {}
            for article_id in article_ids:
                utilities[article_id] = (1 - margins[article_id]) * correlations[article_id]**self.beta

            # sort in descending order and choose first n_subset article ids
            article_ids = list(islice(OrderedDict(sorted(utilities.iteritems(), key=lambda x: x[1], reverse=True)),
                                      self.n_subset))

        ##############################################
        # entropy-correlation-based instance selection
        ##############################################
        elif self.approach in ['entropy-correlation']:
            # run classification to retrieve instance prediction probabilities
            y_prob = self._classify_and_get_probabilities()

            # calculate uncertainty over the whole output prediction distribution (entropy)
            entropies = self._calculate_entropies(y_prob, article_ids)

            # find most centered instances based on cosine similarity
            correlations = self._calculate_correlations(article_ids)

            # calculate utility values
            utilities = {}
            for article_id in article_ids:
                utilities[article_id] = entropies[article_id] * correlations[article_id]**self.beta

            # sort in descending order and choose first n_subset article ids
            article_ids = list(islice(OrderedDict(sorted(utilities.iteritems(), key=lambda x: x[1], reverse=True)),
                                      self.n_subset))

        return article_ids

    @staticmethod
    def plot_curves(n_inner, feature_indices, combinations, metric='fscore'):
        """
        Function plots 5 curves that show how a score given by parameter 'metric' changed through iterations of active
        learning.
        :param feature_indices: List containing single feature space index.
        :param combinations: Dictionary, where single key is the name of the algorithm, and its value is a list with a
        single element defining the parameter of the algorithm.
        :param metric: Determines which metric's score will be displayed. Options: 'ca', 'precison', 'recall', 'fscore'.
        :return:
        """
        idx = feature_indices[0]
        algorithm = combinations.keys()[0]
        param = combinations[algorithm][0]

        approaches = ['random', 'margin', 'entropy', 'margin-correlation', 'entropy-correlation']
        for app in approaches:
            scores = {}
            for i in range(n_inner+1):
                result = DataLoader.load('data/results/model_evaluation/%s/metrics/%d_idx=%d,algorithm=%s,param=%d.csv' %
                                         (app, i, idx, algorithm, param),
                                         delim=';', skip_legend=True, dtype=str)
                if metric == 'ca':
                    scores[i] = result[-4, -1].split()[0]
                elif metric == 'precision':
                    scores[i] = result[-3, -1]
                elif metric == 'recall':
                    scores[i] = result[-2, -1]
                elif metric == 'fscore':
                    scores[i] = result[-1, -1]

            plt.plot(scores.keys(), scores.values(), '-')

        plt.legend(approaches, loc='lower right')
        plt.xlabel('Active learning iteration no.')
        labels = {'ca': 'Classification accuracy', 'precision': 'Precision', 'recall': 'Recall', 'fscore': 'F1-score'}
        plt.ylabel(labels[metric])
        plt.show()

    @staticmethod
    def plot_curves_average(name, n_outer, n_inner, feature_indices, combinations, metric='fscore'):
        idx = feature_indices[0]
        algorithm = combinations.keys()[0]
        param = combinations[algorithm][0]

        approaches = ['random', 'margin', 'entropy', 'margin-correlation', 'entropy-correlation']
        scores = {}
        for app in approaches:
            scores[app] = defaultdict(list)
            for i in range(n_inner+1):
                for j in range(n_outer):
                    result = DataLoader.load('%s/%d/results/model_evaluation/%s/metrics/%d_idx=%d,algorithm=%s,param=%d.csv' %
                                             (name, j, app, i, idx, algorithm, param),
                                             delim=';', skip_legend=True, dtype=str)
                    if metric == 'ca':
                        scores[app][i].append(float(result[-4, -1].split()[0]))
                    elif metric == 'precision':
                        scores[app][i].append(float(result[-3, -1]))
                    elif metric == 'recall':
                        scores[app][i].append(float(result[-2, -1]))
                    elif metric == 'fscore':
                        scores[app][i].append(float(result[-1, -1]))

        for app in approaches:
            avgs = {i: np.mean(scores[app][i]) for i in scores[app]}
            plt.plot(avgs.keys(), avgs.values(), '-')

        plt.legend(approaches, loc='lower right')
        plt.xlabel('Active learning iteration no.')
        labels = {'ca': 'Classification accuracy', 'precision': 'Precision', 'recall': 'Recall', 'fscore': 'F1-score'}
        plt.ylabel(labels[metric])
        plt.show()

    ####################
    # utility functions
    ####################
    def _classify_and_get_probabilities(self):
        clf = Classifier(mode='active_learning', n=self.n, approach=self.approach, skewed_class_weighing=False,
                         feature_indices=self.feature_indices, combinations=self.combinations,
                         seed=self.seed, trace=True)
        results = clf.fit_predict(train=self.features_train, test=self.features_active_learning)
        clf.evaluate_and_save_results()

        # parse results
        idx = self.feature_indices[0]
        algorithm = self.combinations.keys()[0]
        param = self.combinations[algorithm][0]
        y_prob = results[idx][algorithm][param]['prob']

        return y_prob

    @staticmethod
    def _calculate_margins(y_prob, article_ids):
        margins = {}
        for i, prob in enumerate(y_prob):
            prob = sorted(prob, reverse=True)
            margins[article_ids[i]] = prob[0] - prob[1]

        return margins

    @staticmethod
    def _calculate_entropies(y_prob, article_ids):
        entropies = {}
        for i, prob in enumerate(y_prob):
            entropies[article_ids[i]] = entropy(prob, base=2)

        return entropies

    def _calculate_correlations(self, article_ids):
        # calculate cosine similarities between examples
        idx = self.feature_indices[0]
        x = TFIDF(idx).construct(self.features_active_learning).tocsr()
        similarities = 1 - pairwise_distances(x, metric='cosine', n_jobs=-1)

        correlations = {}
        for i in range(similarities.shape[0]):
            correlations[article_ids[i]] = similarities[i, :].sum() / len(article_ids)

        return correlations
