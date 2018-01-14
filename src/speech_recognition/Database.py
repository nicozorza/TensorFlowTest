import numpy as np
import random

class Database:

    def __init__(self, copy=None, batch_size=100):
        if copy is None:
            self.data_arr = []
            self.length = 0
        else:
            self.data_arr = copy
            self.length = len(copy)
        self.batch_size = batch_size

    def append(self, label, data):
        if type(data[0]) is np.float64:
            aux = [label, data]
            self.data_arr.append(aux)
        else:
            for i in range(data.shape[1]):
                aux = [label, data[:, i]]
                self.data_arr.append(aux)
        self.length = len(self.data_arr)

    def get_train_test_set(self, factor):
        n_train = int(factor * self.length)
        aux_data = self.data_arr
        random.shuffle(aux_data)
        test_set = Database(aux_data[:n_train])
        train_set = Database(aux_data[n_train+1:])
        return train_set, test_set

    def get_labels(self):
        labels = []
        for i in range(len(self.data_arr)):
            aux = [0, 0]
            aux[self.data_arr[i][0]] = 1
            labels.append(aux)
        return labels

    def get_data(self):
        data = []
        for i in range(len(self.data_arr)):
            data.append(self.data_arr[i][1])
        return data

    def next_batch(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        data = self.data_arr
        random.shuffle(data)
        return Database(data[:self.batch_size], self.batch_size)

    def print(self):
        return self.data_arr

    def __str__(self):
        return self.data_arr.__str__()

    def __len__(self):
        return self.length
