#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def arg_clf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='reddit', help="name of dataset")
    parser.add_argument('--num_features', type=int, default=1, help="number of features groups")
    parser.add_argument('--num_topics', type=int, default=10, help="number of topics")
    args = parser.parse_args()
    return args


def arg_rnn():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', type=int, default=1000, help="max length of sequences")
    parser.add_argument('--max_num_words', type=int, default=20000, help="max number of words")
    parser.add_argument('-d', '--embedding_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--rnn_units', type=int, default=128, help="units of RNN")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--dense_units', type=int, default=32, help="units of Dense layer")

    parser.add_argument('--dataset', type=str, default='twitter', help="name of dataset")
    embedding_file = '/data/shji/datasets/word2vec/GoogleNews-vectors-negative300.bin'
    parser.add_argument('-f', '--embedding_file', type=str, default=embedding_file, help="embedding file")
    parser.add_argument('--embedding_type', type=str, default='word2vec', help="the type of word embedding")
    parser.add_argument('--valid_split', type=float, default=0.1, help="ratio of validation split")
    parser.add_argument('-act', '--activation', type=str, default='relu', help="type of activation function")
    parser.add_argument('--patience', type=int, default=10, help="number of epochs with no improvement after which training will be stopped")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int ,default=200, help="training epochs")
    args = parser.parse_args()
    return args


def arg_lstm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', type=int, default=1000, help="max length of sequences")
    parser.add_argument('--max_num_words', type=int, default=50000, help="max number of words")
    parser.add_argument('-d', '--embedding_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--lstm_units', type=int, default=128, help="units of LSTM")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--dropout_rate_lstm', type=float, default=0.2, help='dropout rate of lstm unit')
    parser.add_argument('--dense_units', type=int, default=32, help="units of Dense layer")

    parser.add_argument('--dataset', type=str, default='twitter', help="name of dataset")
    embedding_file = '/data/shji/datasets/word2vec/GoogleNews-vectors-negative300.bin'
    parser.add_argument('-f', '--embedding_file', type=str, default=embedding_file, help="embedding file")
    parser.add_argument('--embedding_type', type=str, default='word2vec', help="the type of word embedding")
    parser.add_argument('--valid_split', type=float, default=0.1, help="ratio of validation split")
    parser.add_argument('-act', '--activation', type=str, default='relu', help="type of activation function")
    parser.add_argument('--patience', type=int, default=10, help="number of epochs with no improvement after which training will be stopped")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int ,default=200, help="training epochs")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    a = arg_lstm()
