from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
import config as C
import numpy as np
from sklearn.metrics import f1_score


def _train_model(model, epoch, batch_size, train_x, train_y, val_x, val_y, evaluation='auc', early_stop=5):
    if evaluation == 'acc' or 'recall':
        hist = model.fit(train_x, train_y, batch_size=batch_size, epochs=epoch, validation_data=(val_x, val_y),
                         callbacks=[
                             EarlyStopping(patience=early_stop),
                         ])
        best_score = hist.history['val_acc'][-1]
    elif evaluation == 'f1':
        best_score = -1
        best_weights = None
        best_epoch = 0
        for current_epoch in range(epoch):
            model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
            y_pred = model.predict(val_x, batch_size=batch_size)
            y_pred = np.asarray(y_pred)

            f1 = np.mean([f1_score(val_y[:, j], y_pred[:, j]) for j in range(len(C.Y))])

            print('F1 score is : {}'.format(f1))

            print("Epoch {0} score {1} best_score {2}".format(current_epoch, f1, best_score))

            if f1 > best_score:
                best_score = f1
                best_weights = model.get_weights()
                best_epoch = current_epoch
            else:
                if current_epoch - best_epoch == early_stop:  # early stop
                    break

        model.set_weights(best_weights)
    elif evaluation == 'auc':
        best_score = -1
        best_weights = None
        best_epoch = 0
        for current_epoch in range(epoch):
            model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
            y_pred = model.predict(val_x, batch_size=batch_size)
            total_score = 0

            labels_scores = []
            labels_num = len(C.Y)
            for j in range(labels_num):
                try:
                    score = roc_auc_score(val_y[:, j], y_pred[:, j])
                    total_score += score
                    labels_scores.append(score)
                except ValueError:
                    continue

            print('different labels AUC is : {}'.format(labels_scores))

            total_score /= labels_num

            print("Epoch {0} score {1} best_score {2}".format(current_epoch, total_score, best_score))

            if total_score > best_score:
                best_score = total_score
                best_weights = model.get_weights()
                best_epoch = current_epoch
            else:
                if current_epoch - best_epoch == early_stop:  # early stop
                        break

        model.set_weights(best_weights)

    return model, best_score


def train_folds(X, y, epoch, fold_count, batch_size, get_model_func, evaluation='auc', early_stop=5):
    # skf = StratifiedKFold(n_splits=fold_count, shuffle=False)
    # skf = StratifiedKFold(y, n_folds=fold_count, shuffle=False)

    models = []
    scores = []

    if fold_count <= 1:
        shuffled_indices = np.random.choice(range(len(X)), size=len(X), replace=False)
        X = X[shuffled_indices]
        y = y[shuffled_indices]
        model = get_model_func()
        ratio = 0.95
        split_index = int(len(X) * ratio)
        train_x = X[:split_index]
        train_y = y[:split_index]
        val_x = X[split_index:]
        val_y = y[split_index:]
        model, score = _train_model(model, epoch, batch_size, train_x, train_y, val_x, val_y, evaluation=evaluation,
                                    early_stop=early_stop)
        models.append(model)
        scores.append(score)
    else:
        fold_size = len(X) // fold_count

        for i in range(fold_count):
            fold_start = fold_size * i
            fold_end = fold_size * (i+1)

            if i == fold_count - 1:
                fold_end = len(X)

            train_x = np.concatenate([X[:fold_start], X[fold_end:]])
            train_y = np.concatenate([y[:fold_start], y[fold_end:]])

            val_x = X[fold_start:fold_end]
            val_y = y[fold_start:fold_end]

            print('Running fold {}/{}'.format(i+1, fold_count))
            model = get_model_func()

            model, score = _train_model(model, epoch, batch_size, train_x, train_y, val_x, val_y, evaluation=evaluation,
                                        early_stop=early_stop)

            models.append(model)
            scores.append(score)

    return models, scores
