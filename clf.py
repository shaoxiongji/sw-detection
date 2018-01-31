#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import string
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from tabulate import tabulate
from options import arg_clf
from helpers import load_df, evaluate_prediction


def f_basic(data):
    print("Processing basic features ...")
    num_title_words, num_title_token, num_title_char = [], [], []
    for title in data['tweets']:
        num_title_words.append(len(title.split()))
        tokens = nltk.word_tokenize(title)
        num_title_token.append(len(tokens))
        num_title_char.append(len(title))
    features = {'title_words': num_title_words, 'title_token': num_title_token, 'title_char': num_title_char,}
    return pd.DataFrame(features, columns=['title_words', 'title_token', 'title_char'])


def f_liwc(dataset_name):
    # extracted features using LIWC
    print("Processing LIWC features ...")
    liwc_twitter = pd.read_csv('./data/liwc_features/liwc_{}.csv'.format(dataset_name))
    liwc = liwc_twitter[liwc_twitter.columns[3:]]
    return liwc


def get_all_tags(data):
    print("Processing POS features ...")
    tags_all = []
    for title in data['tweets']:
        tagged_text = nltk.pos_tag(nltk.word_tokenize(title))
        for word, tag in tagged_text:
            if tag not in tags_all:
                tags_all.append(tag)
    return tags_all


def f_pos(data, tags_all):
    tag_dict, tag_count = {}, {}
    for tag in tags_all:
        tag_dict[tag] = 0
        tag_count[tag] = []
    for title in data['tweets']:
        tagged_text = nltk.pos_tag(nltk.word_tokenize(title))
        for word, tag in tagged_text:
            tag_dict[tag]+=1
        for count,tag in zip(tag_dict.values(), tag_dict.keys()):
            tag_count[tag].append(count)
    return pd.DataFrame(tag_count, index=None)


def f_tfidf(data):
    print("Processing TF-IDF features ...")
    X = data['tweets']
    count_vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_features=50)
    X_counts = count_vect.fit_transform(X)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_counts)
    df_tfidf = pd.DataFrame(X_train_tfidf.todense())
    return df_tfidf


def f_topics(data, topic_num):
    def cleaning(article):
        punctuation = set(string.punctuation)
        lemmatize = WordNetLemmatizer()
        one = " ".join([i for i in article.lower().split() if i not in stopwords])
        two = "".join(i for i in one if i not in punctuation)
        three = " ".join(lemmatize.lemmatize(i) for i in two.lower().split())
        return three

    def pred_new(doc):
        one = cleaning(doc).split()
        two = dictionary.doc2bow(one)
        return two

    print("Processing Topics features ...")
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = data['tweets'].map(cleaning)
    text_list = []
    for t in text:
        temp = t.split()
        text_list.append([i for i in temp if i not in stopwords])
    dictionary = corpora.Dictionary(text_list)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
    ldamodel = LdaModel(doc_term_matrix, num_topics=topic_num, id2word = dictionary, passes=50)
    probs = []  # list of probability vectors
    for t in text:
        prob = ldamodel[(pred_new(t))]
        d = dict(prob)
        for i in range(topic_num):
            if i not in d.keys():
                d[i] = 0
        temp = []
        for i in range(topic_num):
            temp.append(d[i])
        probs.append(temp)
    return pd.DataFrame(probs, index=None)


if __name__ == '__main__':
    args = arg_clf()
    df_data = load_df(args.dataset)
    df_data.columns = ['tweets', 'y']
    if args.num_features == 1:
        df_basic = f_basic(df_data)
        df_all = pd.concat([df_basic, df_data['y']], axis=1)
    elif args.num_features == 2:
        df_basic = f_basic(df_data)
        df_tfidf = f_tfidf(df_data)
        df_features = pd.concat([df_basic, df_tfidf], axis=1)
        df_all = pd.concat([df_features, df_data['y']], axis=1)
    elif args.num_features == 3:
        df_basic = f_basic(df_data)
        df_tfidf = f_tfidf(df_data)
        tags_all = get_all_tags(df_data)
        df_pos = f_pos(df_data, tags_all)
        df_features = pd.concat([df_basic, df_tfidf, df_pos], axis=1)
        df_all = pd.concat([df_features, df_data['y']], axis=1)
    elif args.num_features == 4:
        df_basic = f_basic(df_data)
        df_tfidf = f_tfidf(df_data)
        tags_all = get_all_tags(df_data)
        df_pos = f_pos(df_data, tags_all)
        df_topic = f_topics(df_data, args.num_topics)
        df_features = pd.concat([df_basic, df_tfidf, df_pos, df_topic], axis=1)
        df_all = pd.concat([df_features, df_data['y']], axis=1)
    elif args.num_features == 5:
        df_basic = f_basic(df_data)
        df_tfidf = f_tfidf(df_data)
        tags_all = get_all_tags(df_data)
        df_pos = f_pos(df_data, tags_all)
        df_topic = f_topics(df_data, args.num_topics)
        df_liwc = f_liwc(args.dataset)
        df_features = pd.concat([df_basic, df_tfidf, df_pos, df_topic, df_liwc], axis=1)
        df_all = pd.concat([df_features, df_data['y']], axis=1)
    else:
        raise ValueError("Error: number of features groups")

    result_average, h = [], ['Model', 'Acc.', 'Pre.', 'Rec.', 'F1', 'AUC']
    lr_acc, lr_pre, lr_rec, lr_f1, lr_auc = [], [], [], [], []
    rf_acc, rf_pre, rf_rec, rf_f1, rf_auc = [], [], [], [], []
    gbdt_acc, gbdt_pre, gbdt_rec, gbdt_f1, gbdt_auc = [], [], [], [], []
    xgb_acc, xgb_pre, xgb_rec, xgb_f1, xgb_auc = [], [], [], [], []
    if args.dataset == 'Twitter' or args.dataset == 'twitter':
        num_sampling = 5
    else:
        num_sampling = 1
    for i in range(num_sampling):
        if args.dataset == 'Twitter' or args.dataset == 'twitter':
            # under sampling
            df_pos = df_all.loc[df_all['y'] == 1]
            df_neg = df_all.loc[df_all['y'] == 0]
            df_sample = pd.concat([df_pos, df_neg.sample(len(df_pos['y']))])
            df_sample = df_sample.dropna()
            X = df_sample[df_sample.columns[:-1]].as_matrix()
            y = df_sample['y'].as_matrix()
        else:
            df_all = df_all.dropna()
            X = df_all[df_all.columns[:-1]].as_matrix()
            y = df_all['y'].as_matrix()

        # 10-fold cross validation
        num_fold = 10
        kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X):
            num_fold -= 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Logisitc Regression
            clf = LogisticRegression(penalty='l2', tol=1e-6)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:,1]
            acc, pre, rec, f1, auc = evaluate_prediction(y_test, y_pred, k_th=num_fold,
                                                         model_name='Logistic Regression', dataset_name=args.dataset)
            lr_acc.append(acc)
            lr_pre.append(pre)
            lr_rec.append(rec)
            lr_f1.append(f1)
            lr_auc.append(auc)
            # Random Forest
            clf = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            acc, pre, rec, f1, auc = evaluate_prediction(y_test, y_pred, k_th=num_fold,
                                                         model_name='Random Forest', dataset_name=args.dataset)
            rf_acc.append(acc)
            rf_pre.append(pre)
            rf_rec.append(rec)
            rf_f1.append(f1)
            rf_auc.append(auc)
            # GBDT
            clf = GradientBoostingClassifier(max_depth=8, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            acc, pre, rec, f1, auc = evaluate_prediction(y_test, y_pred, k_th=num_fold,
                                                         model_name='GBDT', dataset_name=args.dataset)
            gbdt_acc.append(acc)
            gbdt_pre.append(pre)
            gbdt_rec.append(rec)
            gbdt_f1.append(f1)
            gbdt_auc.append(auc)
            # XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, missing=-999)
            dtest = xgb.DMatrix(X_test, label=y_test, missing=-999)
            params = {'max_depth': 10, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic', 'nthread': -1}
            num_round = 10000
            watchlist = [(dtrain, 'train'), (dtest, 'test')]
            model = xgb.train(params, dtrain, num_round, watchlist, early_stopping_rounds=50, verbose_eval=10)
            y_pred = model.predict(dtest)
            acc, pre, rec, f1, auc = evaluate_prediction(y_test, y_pred, k_th=num_fold,
                                                         model_name='XGBoost', dataset_name=args.dataset)
            xgb_acc.append(acc)
            xgb_pre.append(pre)
            xgb_rec.append(rec)
            xgb_f1.append(f1)
            xgb_auc.append(auc)
    result_average.append(['Logistic Regression', np.mean(lr_acc), np.mean(lr_pre), np.mean(lr_rec), np.mean(lr_f1), np.mean(lr_auc)])
    result_average.append(['Random Forest', np.mean(rf_acc), np.mean(rf_pre), np.mean(rf_rec), np.mean(rf_f1), np.mean(rf_auc)])
    result_average.append(['GBDT', np.mean(gbdt_acc), np.mean(gbdt_pre), np.mean(gbdt_rec), np.mean(gbdt_f1), np.mean(gbdt_auc)])
    result_average.append(['XGB', np.mean(xgb_acc), np.mean(xgb_pre), np.mean(xgb_rec), np.mean(xgb_f1), np.mean(xgb_auc)])
    print(tabulate(result_average, headers=h))
