import pandas as pd
from predicate import is_daily
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
import random


def print_result(metric, _y, _y_hat):
    print('use {}'.format(metric))
    acc, pre, recall = accuracy_score, precision_score, recall_score

    print('accuracy: {} '.format(acc(_y, _y_hat)))
    print('precision: {} '.format(pre(_y, _y_hat)))
    print('recall: {}'.format(recall(_y, _y_hat)))
    print('fbata_score: {}'.format(fbeta_score(_y, _y_hat, beta=1.5)))


file_path = '/Users/kouminquan/Documents/IBM工作/分类测试01.csv'

content = pd.read_csv(file_path, encoding='utf-8', names=['sentence', 'yhat', 'y', 'right', 'acc'], skiprows=[0])

sentence = content['sentence'].tolist()
y = content['y'].astype(int).tolist()

y_hats_0 = [0 if is_daily(s, metric='ensemble') else 1 for s in sentence]
print_result('ensemble', y, y_hats_0)

# y_hats_1 = [0 if is_daily(s, metric='value') else 1 for s in sentence]
# print_result('value', y, y_hats_1)
#
# ensemble
#
# y_hats_2 = [random.choice(Ys) for Ys in zip(y_hats_0, y_hats_1)]
# print_result('ensemble', y, y_hats_2)
