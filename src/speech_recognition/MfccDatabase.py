import numpy as np
import random


class Mfcc:
    def __copy__(self):
        return self

    def __init__(self, mfcc=None, label=None):
        self.mfcc = mfcc
        self.label = label
        if mfcc is None:
            self.n_mfcc = None
            self.n_frames = None
        else:
            self.n_mfcc = np.shape(mfcc)[1]
            self.n_frames = np.shape(mfcc)[0]

    def getNMfcc(self):
        return self.n_mfcc

    def getNFrames(self):
        return self.n_frames

    def __str__(self):
        return 'Label: '+str(self.label)+'\n' + 'Data: '+str(self.mfcc)

    def sampleTrim(self, size):
        if size > self.n_frames:
            print("Size too big")

        center = int(float(self.n_frames)/2)
        return Mfcc(
            mfcc=self.mfcc[center-int(size/2):center+int(size/2)],
            label=self.label
        )



class MfccDatabase:
    def __init__(self, mfccDatabase=None, batch_size=100):
        if mfccDatabase is None:
            self.mfccDatabase = []
        else:
            self.mfccDatabase = mfccDatabase
        self.batch_size = batch_size
        self.length = len(self.mfccDatabase)

    def append(self, mfcc, label):
        self.mfccDatabase.append(Mfcc(mfcc, label))
        self.length = len(self.mfccDatabase)

    def __str__(self):
        aux = ""
        for i in range(len(self.mfccDatabase)):
            aux += str(self.mfccDatabase[i]) + '\n'
        return aux

    def print(self):
        return self.mfccDatabase

    def sampleTrim(self, size=None):

        if size is None:
            aux_size = self.mfccDatabase[0].n_frames
            for i in range(self.length):
                aux2 = self.mfccDatabase[i].n_frames
                if aux2 <= aux_size:
                    aux_size = aux2
            size = aux_size

        aux = MfccDatabase()
        for i in range(self.length):
            trimmed = self.mfccDatabase[i].sampleTrim(size)
            aux.append(trimmed.mfcc, trimmed.label)
        return aux

    def trainTestSet(self, factor):
        n_train = int(factor * self.length)
        aux_data = self.mfccDatabase
        random.shuffle(aux_data)
        train_set = MfccDatabase(aux_data[:n_train])
        test_set = MfccDatabase(aux_data[n_train + 1:])
        return train_set, test_set

    # This method assumes the size is the same for every sample
    def getNMfcc(self):
        if self.mfccDatabase is not []:
            return self.mfccDatabase[0].getNMfcc()
        else:
            return 0

    # This method assumes the size is the same for every sample
    def getNFrames(self):
        if self.mfccDatabase is not []:
            return self.mfccDatabase[0].getNFrames()
        else:
            return 0

    def getData(self):
        data = np.ndarray(
            shape=[self.length, self.getNFrames()*self.getNMfcc()]
        )
        for _ in range(self.length):
            data[_] = np.hstack(self.mfccDatabase[_].mfcc)
        return data


    def getLabels(self):

        labels = np.ndarray(
            shape=[self.length, 10]
        )
        for i in range(len(self.mfccDatabase)):
            aux = np.zeros(10)
            aux[self.mfccDatabase[i].label] = 1
            labels[i] = aux
        return labels

    def next_batch(self, batch_size=100):
        if batch_size is not None:
            self.batch_size = batch_size
        aux = self.mfccDatabase
        random.shuffle(aux)
        data = MfccDatabase(aux[0:batch_size], batch_size)

        return data.getData(), data.getLabels()    #CHEACKEAR ESTO

    def __len__(self):
        return self.length