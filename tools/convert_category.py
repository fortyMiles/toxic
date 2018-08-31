# Created by mqgao at 2018/8/31

"""
Feature: #Enter feature name here
# Enter feature description here

Scenario: #Enter scenario name here
# Enter steps here

Test File Location: # Enter

"""
import pandas as pd


emotion_map = {-2: 'unrelated', 1: 'positive', 0: 'neutral', -1: 'neg'}


def convert_continuos_to_2_categorical(continous_csv, x='comment'):
    columns = continous_csv.columns.tolist()
    assert x in columns, ' x not in columns'

    new_columns = []

    for c in columns:
        for e in emotion_map.values():
            if c.endswith(e):
                new_columns.append(c.replace('_'+e, ''))

    columns = list(set(new_columns))

    assert len(columns) == 20

    columns.append(x)

    merged_result_dict = {k: [] for k in columns}

    columns.remove(x)
    labels = columns

    for ii, row in enumerate(continous_csv.iterrows()):
        if ii % 100 == 0: print('{}/{} '.format(ii, len(continous_csv)), end='')
        content = row[1]
        merged_result_dict['comment'].append(content['comment'])

        for label in labels:
            most_pro_code, most_pro = max([(code, content[label + '_' + emotion])
                                           for code, emotion in emotion_map.items()], key=lambda x: x[1])
            merged_result_dict[label].append(most_pro_code)

    merged_result = pd.DataFrame.from_dict(merged_result_dict)

    return merged_result


def change_n_categorical_to_n_binary_categorical(csv_content, x='comment'):
    labels = csv_content.columns.tolist()
    labels.remove(x)

    binary_columns = [[label + '_unrelated', label + '_positive', label + '_neutral', label + '_neg'] for label in
                      labels]
    binary_columns = [l for category in binary_columns for l in category]
    filed_keys = {k: [] for k in binary_columns}
    filed_keys['comment'] = []

    emotion_map = {-2: 'unrelated', 1: 'positive', 0: 'neutral', -1: 'neg'}

    new_binary_format_comment = pd.DataFrame.from_dict(filed_keys)

    for index, row in enumerate(csv_content.iterrows()):
        content = row[1]

        one_rating = {k: 0 for k in filed_keys}

        for label in labels:
            rate = content[label]
            key = label + '_' + emotion_map[rate]
            assert key in one_rating
            one_rating[key] = 1

        one_rating['comment'] = content['comment']

        new_binary_format_comment = new_binary_format_comment.append(one_rating, ignore_index=True)

    return new_binary_format_comment




