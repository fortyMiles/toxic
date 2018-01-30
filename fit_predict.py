from toxic.model import get_model
from toxic.train_utils import train_folds
from tools.dataset_untils import get_train_test_and_embedding

import argparse
import numpy as np
import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6'

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("--embedding-path", default=None)
    parser.add_argument("--result-path", default="toxic_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sentences-length", type=int, default=500)
    parser.add_argument("--recurrent-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--dense-size", type=int, default=32)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument('--epoch', type=int, default=5)

    args = parser.parse_args()

    if args.fold_count <= 1:
        raise ValueError("fold-count should be more than 1")

    print("Loading data...")

    X_train, y_train, X_test, embedding_matrix = get_train_test_and_embedding(args.train_file_path,
                                                                       args.test_file_path,
                                                                       args.sentences_length,
                                                                       args.embedding_path)

    get_model_func = lambda: get_model(
        embedding_matrix,
        args.sentences_length,
        args.dropout_rate,
        args.recurrent_units,
        args.dense_size)

    print("Starting to train models...")
    model, hist = train_folds(X_train, y_train, args.epoch, args.batch_size, get_model_func)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    print("Predicting results...")
    y_pred = model.predict(X_test)

    submission = pd.read_csv('data/sample_submission.csv')
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred / 1.4

    submission.head()

    val_acc = np.mean(hist.history['val_acc'])
    submission.to_csv('submission_lstm_{}.csv'.format(val_acc), index=False)

    # test_predicts_list = []
    # for fold_id, model in enumerate(models):
    #     model_path = os.path.join(args.result_path, "model{0}_weights.npy".format(fold_id))
    #     np.save(model_path, model.get_weights())
    #
    #     test_predicts_path = os.path.join(args.result_path, "test_predicts{0}.npy".format(fold_id))
    #     test_predicts = model.predict(X_test, batch_size=args.batch_size)
    #     test_predicts_list.append(test_predicts)
    #     np.save(test_predicts_path, test_predicts)
    #
    # test_predicts = np.ones(test_predicts_list[0].shape)
    # for fold_predict in test_predicts_list:
    #     test_predicts *= fold_predict
    #
    # test_predicts **= (1. / len(test_predicts_list))
    # test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT
    #
    # test_ids = test_data["id"].values
    # test_ids = test_ids.reshape((len(test_ids), 1))
    #
    # test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    # test_predicts["id"] = test_ids
    # test_predicts = test_predicts[["id"] + CLASSES]
    # submit_path = os.path.join(args.result_path, "submit")
    # test_predicts.to_csv(submit_path, index=False)

if __name__ == "__main__":
    main()
