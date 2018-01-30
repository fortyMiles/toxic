import pandas as pd
import numpy as np
import os
import argparse
from tools.initial_train_test_data import get_train_test_and_embedding
from fasttext_bn.model import get_model
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("--embedding-path", type=str, default=None)

    args = parser.parse_args()

    max_features = 50000
    maxlen = 150
    batch_size = 32
    epochs = 100

    if args.embedding_path is None:
        embedding_path = None
        embedding_dim = 64
    else:
        embedding_path = args.embedding_path
        embedding_dim = None

    X_train, y_train, X_test, comment_embedding = get_train_test_and_embedding(
        args.train_file_path, args.test_file_path, maxlen, max_features, embedding_path, embedding_dim
    )

    model = get_model(comment_embedding, sequence_length=maxlen, vocab_size=max_features, embedding_dim=embedding_dim)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
        TensorBoard(log_dir='./logs', batch_size=batch_size)
    ]

    hist = model.fit(X_train, y_train, batch_size=batch_size,
                     epochs=epochs, validation_split=0.1,
                     callbacks=callbacks)
    y_pred = model.predict(X_test)

    submission = pd.read_csv('data/sample_submission.csv')
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred / 1.4

    submission.head()

    val_acc = np.mean(hist.history['val_acc'])
    submission.to_csv('submission_bn_fasttext_{}.csv'.format(val_acc), index=False)


if __name__ == '__main__':
    main()
