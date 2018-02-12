#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from sklearn.model_selection import KFold
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from tqdm import tqdm
from tabulate import tabulate
from options import arg_lstm
from helpers import load_data, evaluate_prediction


def build_LSTM(args, mat_embedding):
    embedding_layer = Embedding(nb_words, args.embedding_dim, weights=[mat_embedding],
                                input_length=args.max_seq_len, trainable=False)
    lstm_layer = LSTM(args.lstm_units, dropout=args.dropout_rate_lstm, recurrent_dropout=args.dropout_rate)
    sequence_1_input = Input(shape=(args.max_seq_len,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)
    sequence_2_input = Input(shape=(args.max_seq_len,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    x2 = lstm_layer(embedded_sequences_2)
    merged = concatenate([x1, x2])
    merged = Dropout(args.dropout_rate)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(args.dense_units, activation='relu')(merged)
    merged = Dropout(args.dropout_rate)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


if __name__ == '__main__':
    args_lstm = arg_lstm()
    X1, X2, y = load_data(dataset_name=args_lstm.dataset)
    all_text = []
    for i in range(len(X1)):
        all_text.append(X1[i])
        all_text.append(X2[i])

    tokenizer = Tokenizer(num_words=args_lstm.max_num_words)
    tokenizer.fit_on_texts(all_text)
    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    print('Loading pertrained embedding file')
    word2vec = KeyedVectors.load_word2vec_format(args_lstm.embedding_file, binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    print('Preparing embedding matrix')
    nb_words = min(args_lstm.max_num_words, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, args_lstm.embedding_dim))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    h, result = ['Model', 'Acc.', 'Pre.', 'Rec.', 'F1', 'AUC'], []
    list_acc, list_pre, list_rec, list_f1, list_auc = [], [], [], [], []

    # 10 fold
    num_fold = 10
    kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)
    for train_ix, test_ix in tqdm(kf.split(y)):
        title_train = X1[train_ix]
        title_test = X1[test_ix]
        usertext_train = X2[train_ix]
        usertext_test = X2[test_ix]
        y_train = y[train_ix]
        y_test = y[test_ix]

        tokenized_title_train = tokenizer.texts_to_sequences(title_train)
        tokenized_title_test = tokenizer.texts_to_sequences(title_test)
        tokenized_usertext_train = tokenizer.texts_to_sequences(usertext_train)
        tokenized_usertext_test = tokenizer.texts_to_sequences(usertext_test)
        X1_train = sequence.pad_sequences(tokenized_title_train, maxlen=args_lstm.max_seq_len, dtype='float64')
        X1_test = sequence.pad_sequences(tokenized_title_test, maxlen=args_lstm.max_seq_len, dtype='float64')
        X2_train = sequence.pad_sequences(tokenized_usertext_train, maxlen=args_lstm.max_seq_len, dtype='float64')
        X2_test = sequence.pad_sequences(tokenized_usertext_test, maxlen=args_lstm.max_seq_len, dtype='float64')

        # train model and predict
        model = build_LSTM(args=args_lstm, mat_embedding=embedding_matrix)

        early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
        model.fit([X1_train, X2_train], y_train, batch_size=args_lstm.batch_size, epochs=args_lstm.epochs,
                  validation_split=0.1, callbacks=[early])

        preds = model.predict([X1_test, X2_test])
        y_pred = preds.ravel()
        acc, pre, rec, f1, auc = evaluate_prediction(y_test, y_pred, k_th=num_fold, model_name='LSTM',
                                                     dataset_name=args_lstm.dataset)

        list_acc.append(acc)
        list_pre.append(pre)
        list_rec.append(rec)
        list_f1.append(f1)
        list_auc.append(auc)
        result.append(['LSTM-word2vec', acc, pre, rec, f1, auc])
        num_fold += 1
    result.append(['average', np.mean(list_acc), np.mean(list_pre),
               np.mean(list_rec), np.mean(list_f1), np.mean(list_auc)])
    print(tabulate(result, headers=h))

