from functools import wraps
import pickle
import os
import time
import hashlib


def get_pickled_name(args, kwargs):
    return str(args) + str(kwargs) + '.pickle'


def load_from_pickle(func):
    @wraps(func)
    def _wrap(*args, **kwargs):
        cache_path = 'cache/'
        pickle_name = get_pickled_name(args, kwargs)
        hash_str = hashlib.md5(pickle_name.encode()).hexdigest()
        pickle_name = os.path.join(cache_path, hash_str + '.pickle')

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        if os.path.exists(pickle_name):
            with open(pickle_name, 'rb') as f:
                print('exits pickle file, load pickled data')
                dump_data = pickle.load(f)
                result = dump_data['result']
        else:
            result = func(*args, **kwargs)
            dump_data = {'result': result, 'args': str(args) + str(kwargs)}
            with open(pickle_name, 'wb') as f:
                pickle.dump(dump_data, f)
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


