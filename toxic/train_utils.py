from sklearn.metrics import roc_auc_score
import numpy as np


def _train_model(model, epoch, batch_size, train_x, train_y, val_x, val_y):
    best_score = -1
    best_weights = None
    best_epoch = 0

    for current_epoch in range(epoch):
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)
        total_score = 0

        for j in range(6):
            score = roc_auc_score(val_y[:, j], y_pred[:, j])
            total_score += score

            total_score /= 6.

        if total_score > best_score:
            best_score = total_score
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 5:  # early stop
                    break

        print("Epoch {0} score {1} best_score {2}".format(current_epoch, total_score, best_score))

    model.set_weights(best_weights)
    return model, best_score


def train_folds(X, y, epoch, fold_count, batch_size, get_model_func):
    # skf = StratifiedKFold(n_splits=fold_count, shuffle=False)
    # skf = StratifiedKFold(y, n_folds=fold_count, shuffle=False)

    fold_size = len(X) // fold_count

    models = []
    scores = []
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

        model, score = _train_model(model, epoch, batch_size, train_x, train_y, val_x, val_y)

        models.append(model)
        scores.append(score)

    return models
