#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import csv
import numpy as np
import pandas as pd
import datetime
from sklearn import metrics


def load_df(dataset_name):
    """
    Load data as DataFrame for logistic regression and ensemble models, without sampling
    :param dataset_name: name of dataset
    :return: DataFrame
    """
    print('Loading dataset DataFrame')
    if dataset_name == 'EP' or dataset_name == 'ep':
        file_name = './data/experience_project/ep.csv'
        df = pd.read_csv(file_name)
        return df
    elif dataset_name == 'twitter' or dataset_name == 'Twitter':
        file_name = './data/twitter/twitter.xlsx'
        df = pd.read_excel(file_name)
        return df
    elif dataset_name in ['popular', 'all', 'AskReddit', 'books', 'gaming', 'movies', 'Jokes']:
        file_name = './data/reddit/{}.csv'.format(dataset_name)
        df = pd.read_csv(file_name)
        return df
    else:
        raise ValueError("Error: unrecognized dataset")


def load_data(dataset_name):
    """
    Load data for NN models, with under sampling from Twitter dataset
    :param dataset_name: name of dataset
    :return: X, y
    """
    print('Loading text dataset')
    if dataset_name == 'EP' or dataset_name == 'ep':
        file_name = './data/experience_project/ep.csv'
        df = pd.read_csv(file_name)
        X = df['body'].as_matrix()
        y = df['y'].as_matrix()
        return X, y
    elif dataset_name == 'twitter' or dataset_name == 'Twitter':
        file_name = './data/twitter/twitter.xlsx'
        df = pd.read_excel(file_name)
        df = df.dropna()
        df_pos = df.loc[df['y'] == 1]
        df_neg = df.loc[df['y'] == 0]
        df = pd.concat([df_pos, df_neg.sample(len(df_pos['y']))])
        df = df.sample(frac=1)
        X = df['tweets'].as_matrix()
        y = df['y'].as_matrix()
        return X, y
    elif dataset_name in ['popular', 'all', 'AskReddit', 'books', 'gaming', 'movies', 'Jokes']:
        file_name = './data/reddit/{}.csv'.format(dataset_name)
        df = pd.read_csv(file_name)
        X1 = df['title'].as_matrix()
        X2 = df['usertext'].as_matrix()
        y = df['y'].as_matrix()
        return X1, X2, y
    else:
        raise ValueError("Error: unrecognized dataset")


def find_threshold(fpr, tpr, threshold):
    rate = np.array(tpr) + np.array(fpr)
    return threshold[np.argmax(rate)]


def evaluate_prediction(y_test, y_pred, k_th, model_name, dataset_name):
    fpr, tpr, th = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    df_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=None)
    df_results.to_csv('./save/prediction_{}_{}_{}.csv'.format(dataset_name, model_name, k_th), index=False)

    o_threshold = 0.5
    for i in range(len(y_pred)):
        if y_pred[i] >= o_threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    acc = metrics.accuracy_score(y_test, y_pred)
    pre = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    dict_eval = {'date': datetime.date.today(),
                 'model': model_name,
                 'accuracy': acc,
                 'precision': pre,
                 'recall': rec,
                 'f-score': f1,
                 'roc': roc_auc,
                 'note': '{}_th fold'.format(k_th),
                 'dataset': dataset_name
                 }
    with open('./output/{}.csv'.format(dataset_name), 'a') as f:
        field_names = ['date', 'model', 'accuracy', 'precision', 'recall', 'f-score', 'roc', 'note', 'dataset']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writerow(dict_eval)
    return acc, pre, rec, f1, roc_auc
