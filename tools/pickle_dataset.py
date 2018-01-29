from functools import wraps
import pickle
import os
import time


def get_pickled_name(args, kwargs):
    return str(args) + str(kwargs) + '.pickle'


def load_from_pickle(func):
    @wraps(func)
    def _wrap(*args, **kwargs):
        pickle_name = get_pickled_name(args, kwargs)
        if os.path.exists(pickle_name):
            with open(pickle_name, 'rb') as f:
                print('exits pickle file, load pickled data')
                result = pickle.load(f)
        else:
            result = func(*args, **kwargs)
            with open(pickle_name, 'wb') as f:
                pickle.dump(result, f)
        return result
    return _wrap


@load_from_pickle
def mock_generate(min_num, max_num):
    result = []

    for i in range(min_num, max_num):
        result.append(i)
        time.sleep(0.1)
    return result


if __name__ == '__main__':
    begin = time.time()
    L = mock_generate(0, 10)
    end = time.time()

    assert 1.0 < end - begin < 1.1
    assert L == [i for i in range(0, 10)]

    begin = time.time()
    L = mock_generate(0, 10)
    end = time.time()

    assert end - begin < 0.1
    assert L == [i for i in range(0, 10)]

    print('test end!')


