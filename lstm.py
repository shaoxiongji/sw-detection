#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, GlobalMaxPool1D, Dropout, BatchNormalization
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping

from options import arg_lstm
from helpers import load_data, evaluate_prediction


def build_LSTM(args):
    inp = Input(shape=(args.max_seq_len, ))
    x = Embedding(input_dim=args.max_num_words, output_dim=args.embedding_dim)(inp)
    x = LSTM(units=args.lstm_units, activation='tanh', dropout=args.dropout_rate, return_sequences=True)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(args.dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(args.dense_units, activation="relu")(x)
    x = Dropout(args.dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # parse parameters
    args_lstm = arg_lstm()
    if args_lstm.dataset == 'twitter' or args_lstm.dataset == 'Twitter':
        n_sampling = 5
    else:
        n_sampling = 1
    h, result = ['Model', 'Acc.', 'Pre.', 'Rec.', 'F1', 'AUC'], []
    list_acc, list_pre, list_rec, list_f1, list_auc = [], [], [], [], []
    while n_sampling > 0:
        # load data
        X, y = load_data(dataset_name=args_lstm.dataset)

        # split train test
        kfold = KFold(n_splits=10, shuffle=True, random_state=1234)
        num_fold = 0
        for train_ix, test_ix in tqdm(kfold.split(y)):
            list_sentences_train = X[train_ix]
            y_train = y[train_ix]
            list_sentences_test = X[test_ix]
            y_test = y[test_ix]

            tokenizer = text.Tokenizer(num_words=args_lstm.max_num_words)
            tokenizer.fit_on_texts(list(list_sentences_train))
            list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
            list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
            X_train = sequence.pad_sequences(list_tokenized_train, maxlen=args_lstm.max_seq_len, dtype='float64')
            X_test = sequence.pad_sequences(list_tokenized_test, maxlen=args_lstm.max_seq_len, dtype='float64')

            # train model and predict
            model = build_LSTM(args=args_lstm)

            early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
            model.fit(X_train, y_train, batch_size=args_lstm.batch_size, epochs=args_lstm.epochs,
                      validation_split=args_lstm.valid_split, callbacks=[early])

            y_pred = model.predict(X_test).reshape(y_test.shape)

            acc, pre, rec, f1, auc = evaluate_prediction(y_test, y_pred, k_th=num_fold, model_name='LSTM',
                                                         dataset_name=args_lstm.dataset)

            list_acc.append(acc)
            list_pre.append(pre)
            list_rec.append(rec)
            list_f1.append(f1)
            list_auc.append(auc)
            result.append(['LSTM', acc, pre, rec, f1, auc])
            num_fold += 1
        n_sampling -= 1
    result.append(['average', np.mean(list_acc), np.mean(list_pre),
                   np.mean(list_rec), np.mean(list_f1), np.mean(list_auc)])
    print(tabulate(result, headers=h))

