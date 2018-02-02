from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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

            print("Epoch {0} score {1} best_score {2}".format(current_epoch, total_score, best_score))

            if total_score > best_score:
                best_score = total_score
                best_weights = model.get_weights()
                best_epoch = current_epoch
            else:
                if current_epoch - best_epoch == 5:  # early stop
                    break

    model.set_weights(best_weights)
    return model, best_score


def train_folds(X, y, epoch, fold_count, batch_size, get_model_func):
    skf = StratifiedKFold(n_splits=fold_count, shuffle=False)
    # skf = StratifiedKFold(y, n_folds=fold_count, shuffle=False)

    models = []
    scores = []
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        print('Running fold {}/{}'.format(i+1, fold_count))
        model = get_model_func()

        model, score = _train_model(model, epoch, batch_size, X[train_indices], y[train_indices], X[test_indices], y[test_indices])

        models.append(model)
        scores.append(score)

    return models
