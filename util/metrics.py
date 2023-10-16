from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def corrcoef(a, b):
    return np.corrcoef(a, b)[0][1]


def mae(a, b):
    return mean_absolute_error(a, b)

def mse(a, b):
    return mean_squared_error(a, b)


def rmse(a, b):
    return np.sqrt(mse(a, b))


def first(a, b):
    return a


def second(a, b):
    return b


class Metricser:
    def __init__(self, metrics_dict=None):
        if metrics_dict == None:
            self.metrics_dict = {
                'PC': corrcoef,
                'MAE': mae,
                'RMSE': rmse,
                # 'first': first,
            }
        else:
            self.metrics_dict = metrics_dict

    def get_metrics(self):
        return list(self.metrics_dict.keys())

    def __call__(self, a, b):
        res_dict = {}
        for k, f in self.metrics_dict.items():
            res_dict[k] = f(a, b)
        return res_dict


class MetricsList:
    def __init__(self, len, init_value=0, keys=None):
        self.arr = []
        for i in range(len):
            self.arr.append(defaultdict(lambda: init_value))
        if keys is not None:
            for i in range(len):
                for k in keys:
                    self.arr[i][k] = init_value

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return self.arr[idx]

    def __setitem__(self, idx, value):
        self.update(idx, value)

    def custom_update(self, idx, val, op_fn):
        target_dict = self.arr[idx]
        if isinstance(val, dict):
            for k, v in val.items():
                target_dict[k] = op_fn(target_dict[k], v)
        else:
            for k, v in target_dict.items():
                target_dict[k] = op_fn(v, val)

    def custom_update_all(self, val, op_fn):
        for i in range(len(self)):
            self.custom_update(i, val, op_fn)

    def unpack(self):
        res_dict = defaultdict(list)
        for i in range(len(self)):
            for k, v in self.arr[i].items():
                res_dict[k].append(v)
                assert len(res_dict[k]) == i + 1
        return res_dict

    def update(self, idx, val):
        self.custom_update(idx, val, lambda a, b: b)

    def add(self, idx, val):
        self.custom_update(idx, val, lambda a, b: a + b)

    def dive(self, idx, val):
        self.custom_update(idx, val, lambda a, b: a / b)

    def dive_all(self, val):
        self.custom_update_all(val, lambda a, b: a / b)


if __name__ == '__main__':
    # m = Metricser({'F': first, 'S': second})
    m = Metricser()
    mlist = MetricsList(4, 0)

    print(mlist.unpack())

    a = [1, 2, 3]
    b = [4, 5, 6]
    print('pc', corrcoef(a, b))
    print('rmse', rmse(a, b))
    print('mse', mse(a, b))

    mlist.add(0, {'PC':0, 'RMSE': 0, 'MSE':0})
    for i in range(1, 4):
        mlist.add(i, m(a, b))
    print(mlist.unpack())
    for i in range(4):
        mlist.add(i, 4)
    print(mlist.unpack())
    for i in range(4):
        mlist.add(i, m(a, b))
    print(mlist.unpack())
