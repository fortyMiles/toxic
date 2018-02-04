from toxic.model import get_model
from toxic.train_utils import train_folds
from tools.initial_train_test_data import get_train_test_and_embedding
import datetime
import re

import argparse
import numpy as np
import os
import pandas as pd
from tools.utilities import softmax

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# Don't pre-allocate memory; allocate as-needed

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=tf_config))

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4
CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument('--test-mode', type=bool, default=False)
    parser.add_argument("--embedding-path", default=None)
    parser.add_argument("--result-path", default="toxic_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sentences-length", type=int, default=500)
    parser.add_argument("--recurrent-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--dense-size", type=int, default=32)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--load-pretrained', type=bool, default=False)

    args = parser.parse_args()

    if args.fold_count <= 1:
        raise ValueError("fold-count should be more than 1")

    print("Loading data...")

    vocab_size = 50000
    embedding_dim = 64

    X_train, y_train, X_test, embedding_matrix = get_train_test_and_embedding(
        train_csv=args.train_file_path, test_csv=args.test_file_path,
        sequence_length=args.sentences_length, vocab_size=vocab_size,
        embedding_file=args.embedding_path, embedding_dim=embedding_dim)

    test_data = pd.read_csv(args.test_file_path)
    test_ids = test_data["id"].values

    get_model_func = lambda: get_model(
        embedding_matrix,
        args.sentences_length,
        args.dropout_rate,
        args.recurrent_units,
        args.dense_size)

    if args.test_mode is True:
        print('in test mode!')
        X_size = X_train.shape[0]
        sub_train_indices = np.random.choice(range(X_size), size=int(X_size*0.1), replace=False)
        # X_test_size = X_test.shape[0]
        # sub_test_indices = np.random.choice(range(X_test_size), size=int(X_test_size*0.1), replace=False)
    else:
        print(' not in test mode')
        X_size = X_train.shape[0]
        sub_train_indices = np.random.choice(range(X_size), size=int(X_size*1), replace=False)
        # X_test_size = X_test.shape[0]
        # sub_test_indices = np.random.choice(range(X_test_size), size=int(X_test_size*1), replace=False)

    X_train = X_train[sub_train_indices]
    y_train = y_train[sub_train_indices]

    # X_test = X_test[sub_test_indices]
    # test_ids = test_ids[sub_test_indices]

    print("Starting to train models...")
    # model, hist = train_folds(X_train, y_train, args.epoch, args.batch_size, get_model_func)

    if args.load_pretrained is False:
        models, scores = train_folds(
            X=X_train, y=y_train,
            epoch=args.epoch, fold_count=args.fold_count,
            batch_size=args.batch_size,
            get_model_func=get_model_func)
    else:
        models = [get_model_func() for _ in range(args.fold_count)]
        scores = [0.98729337078270507, 0.98741052541081709, 0.98875435673905765, 0.98888426426103615, 0.98921313005210798, 0.98900847797211722, 0.98849751432495514, 0.98839252464184923, 0.98844750778401702, 0.98633935039731879]

    print('the fold score is : {}'.format(scores))
    validation_scores = np.mean(scores)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    # print("Predicting results...")
    # y_pred = model.predict(X_test)
    #

    probabilities = softmax(scores)
    # test_predicts_list = []
    # for fold_id, model in enumerate(models):
    test_predicts = np.zeros(shape=(X_test.shape[0], len(CLASSES)))
    fold_id = 0
    print('predicate test set!')
    for model, prob in zip(models, probabilities):
        model_path = os.path.join(args.result_path, 'model{0}_weights.npy'.format(fold_id))

        if args.load_pretrained is True:
            weights = np.load('model{0}_weights.npy')
            model.set_weights(weights)
        else:
            np.save(model_path, model.get_weights())

        # test_predicts_path = os.path.join(args.result_path, "test_predicts{0}.npy".format(fold_id))
        t = model.predict(X_test, batch_size=args.batch_size)
        test_predicts += prob * t
        # test_predicts_list.append(test_predicts)
        # np.save(test_predicts_path, test_predicts)

    # test_predicts = np.ones(test_predicts_list[0].shape)
    # for fold_predict in test_predicts_list:
    #     test_predicts += fold_predict

    # test_predicts /= len(test_predicts_list)

    # test_predicts **= (1. / len(test_predicts_list))
    # test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT
    #

    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    embedding_size = re.findall('[(cbow)|(skip)]-(\d+)-',args.embedding_path)[0]
    parameters = "emb-{}-batch_size-{}-sen_len-{}-RUNIT-{}-dense_s-{}".format(
        embedding_size,
        args.batch_size,
        args.sentences_length,
        args.recurrent_units,
        args.dense_size
    )

    submit_path = os.path.join(args.result_path, "{}_{}_submission_lstm_{}.csv".format(parameters, now, validation_scores))
    test_predicts.to_csv(submit_path, index=False)


if __name__ == "__main__":
    main()
