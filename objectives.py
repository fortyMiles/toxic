import tensorflow as tf
import numpy as np

from keras import backend as K
from keras.layers import Lambda

_epsilon = K.epsilon()


def roc_auc_score(y_pred, y_true):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """

    # pos = y_true[K.cast(y_true, bool)]
    # neg = y_pred[~K.cast()]
    with K.name_scope("RocAucScore"):

        # y_true = K.cast(y_true, 'float32')
        # y_pred = K.cast(y_pred, 'float32')
        mask = K.greater(y_true, 0)

        pos = y_pred[mask]
        neg = Lambda(lambda x: x * ~mask)(y_pred)

        pos = K.expand_dims(pos, 0)
        neg = K.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = K.zeros_like(pos * neg) + pos - neg - gamma

        masked = Lambda(lambda x: x * (difference < 0.0))(difference)
        # masked = K.boolean_mask(difference, difference < 0.0)

        return K.sum(tf.pow(-masked, p))
