from sklearn.metrics import log_loss
from keras.callbacks import EarlyStopping
import numpy as np


def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    # while current_epoch < 5:
    # model.fit(train_x, train_y, batch_size=batch_size, epochs=1, validation_split=0.1)
    # y_pred = model.predict(val_x, batch_size=batch_size)
    #
    #     total_loss = 0
    #     for j in range(6):
    #         loss = log_loss(val_y[:, j], y_pred[:, j])
    #         total_loss += loss
    #
    #     total_loss /= 6.
    #
    #     print("Epoch {0} loss {1} best_loss {2}".format(current_epoch, total_loss, best_loss))
    #
    #     current_epoch += 1
    #     if total_loss < best_loss or best_loss == -1:
    #         best_loss = total_loss
    #         best_weights = model.get_weights()
    #         best_epoch = current_epoch
    #     else:
    #         if current_epoch - best_epoch == 5:
    #             break

    model.set_weights(best_weights)
    return model


def train_folds(X, y, epoch, batch_size, get_model_func):
    # fold_size = len(X) // fold_count
    # models = []

    model = get_model_func()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    ]

    hist = model.fit(X, y, batch_size=batch_size,
                     epochs=epoch,
                     validation_split=0.1,
                     callbacks=callbacks
                     )
    # for epoch in range(max_epoch):
    # for fold_id in range(0, fold_count):
    #     fold_start = fold_size * fold_id
    #     fold_end = fold_start + fold_size
    #
    #     if fold_id == fold_size - 1:
    #         fold_end = len(X)
    #
    #     train_x = np.concatenate([X[:fold_start], X[fold_end:]])
    #     train_y = np.concatenate([y[:fold_start], y[fold_end:]])
    #
    #     val_x = X[fold_start:fold_end]
    #     val_y = y[fold_start:fold_end]
    #     model = _train_model(get_model_func(), batch_size, X, y, None, None)
    #     models.append(model)

    return model, hist
