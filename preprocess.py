import jieba
import pandas as pd


jieba.load_userdict('dic.txt')


def cut(string): return ' '.join(jieba.cut(string))


target_fields = ['comment_text']
root_path = '/Users/Minchiuan/Workspace/bank'
train_path, test_path = '{}/train.csv'.format(root_path), '{}/test.csv'.format(root_path)
submission = '{}/sample_submission.csv'.format(root_path)


def apply_func_to_columns(csv_file_name, fields, func):
    csv_file = pd.read_csv(csv_file_name, encoding='utf-8')
    for f in fields:
        csv_file[f] = csv_file[f].apply(lambda s: func(s))
    csv_file.to_csv(csv_file_name, index=False)
    print('done!')


for f in [train_path, test_path, submission]:
    apply_func_to_columns(f, target_fields, cut)



