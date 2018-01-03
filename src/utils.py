from __future__ import print_function
import numpy as np

def save_as_wv_format(filename, data):
    with open(filename, 'w') as f:
        if type(data) == list:
            embedding_size = data[0].shape[1]
            nums_type = [data[i].shape[0] for i in range(3)]
            print(sum(nums_type), embedding_size, file=f)
            shift = [0] + np.cumsum(nums_type).tolist()
            for i in range(3):
                for j in range(nums_type[i]):
                    print(shift[i]+j, *data[i][j], file=f)
        else:
            nums, embedding_size = data.shape
            print(nums, embedding_size, file=f)
            for j in range(nums):
                print(j, *data[j], file=f)

def load_from_wv_format(filename):
    with open(filename) as f:
        l = f.readline().split()
        total_num, embedding_size = int(l[0]), int(l[1])
        res = np.zeros((total_num, embedding_size), dtype=float)
        ls = map(lambda x: x.strip().split(), f.readlines())
        for line in ls:
            res[int(line[0])] = list(map(float, line[1:]))
    return res
