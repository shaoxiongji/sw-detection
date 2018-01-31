#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from tabulate import tabulate
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from tqdm import tqdm

from options import arg_lstm
from helpers import load_data, evaluate_prediction


def build_LSTM(args):
    # define the model structure
    embedding_layer = Embedding(nb_words, args.embedding_dim, weights=[embedding_matrix],
                                input_length=args.max_seq_len, trainable=False)
    lstm_layer = LSTM(args.lstm_units, activation='tanh', dropout=args.dropout_rate)
    sequence_input = Input(shape=(args.max_seq_len,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = lstm_layer(embedded_sequences)
    x = Dropout(args.dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(args.dense_units, activation='relu')(x)
    x = Dropout(args.dropout_rate)(x)
    x = BatchNormalization()(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


if __name__ == "__main__":
    # parse parameters
    args_lstm = arg_lstm()

    print('Loading pertrained embedding file')
    word2vec = KeyedVectors.load_word2vec_format(args_lstm.embedding_file, binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    X, y = load_data(dataset_name=args_lstm.dataset)
    all_text = []
    for i in range(len(X)):
        all_text.append(X[i])

    tokenizer = Tokenizer(num_words=args_lstm.max_num_words)
    tokenizer.fit_on_texts(all_text)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))
    print('Preparing embedding matrix')
    nb_words = min(args_lstm.max_num_words, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, args_lstm.embedding_dim))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    if args_lstm.dataset == 'twitter' or args_lstm.dataset == 'Twitter':
        n_sampling = 5
    else:
        n_sampling = 1
    h, result = ['Model', 'Acc.', 'Pre.', 'Rec.', 'F1', 'AUC'], []
    list_acc, list_pre, list_rec, list_f1, list_auc = [], [], [], [], []
    while n_sampling > 0:
        X, y = load_data(dataset_name=args_lstm.dataset)

        # 10 fold
        num_fold = 10
        kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)

        h, result = ['Model', 'Acc.', 'Pre.', 'Rec.', 'F1', 'AUC'], []
        list_acc, list_pre, list_rec, list_f1, list_auc = [], [], [], [], []
        for train_index, test_index in tqdm(kf.split(X)):
            num_fold -= 1
            text_train, text_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_text_seq = tokenizer.texts_to_sequences(text_train)
            test_text_seq = tokenizer.texts_to_sequences(text_test)

            train_tweet = pad_sequences(train_text_seq, maxlen=args_lstm.max_seq_len)
            test_tweet = pad_sequences(test_text_seq, maxlen=args_lstm.max_seq_len)
            train_labels = np.array(y_train)
            test_labels = np.array(y_test)

            # sample train/validation data
            np.random.seed(1234)
            perm = np.random.permutation(len(train_tweet))
            idx_train = perm[:int(len(train_tweet) * (1 - args_lstm.valid_split))]
            idx_val = perm[int(len(train_tweet) * (1 - args_lstm.valid_split)):]
            data_train = train_tweet[idx_train]
            labels_train = train_labels[idx_train]
            data_val = train_tweet[idx_val]
            labels_val = train_labels[idx_val]

            # train the model
            model = build_LSTM(args=args_lstm)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            hist = model.fit(data_train, labels_train, validation_data=(data_val, labels_val),
                             epochs=args_lstm.epochs, batch_size=args_lstm.batch_size, shuffle=True, callbacks=[early_stopping])
            # predict
            print('Testing')
            preds = model.predict(test_tweet, batch_size=32, verbose=1)
            y_pred = preds.ravel()
            acc, pre, rec, f1, auc = evaluate_prediction(y_test=y_test, y_pred=y_pred,
                                                         k_th=num_fold, model_name='LSTM-word2vec', dataset_name=args_lstm.dataset)
            list_acc.append(acc)
            list_pre.append(pre)
            list_rec.append(rec)
            list_f1.append(f1)
            list_auc.append(auc)
            result.append(['LSTM-word2vec', acc, pre, rec, f1, auc])
        n_sampling -= 1
    result.append(['average', np.mean(list_acc), np.mean(list_pre),
                    np.mean(list_rec), np.mean(list_f1), np.mean(list_auc)])
    print(tabulate(result, headers=h))