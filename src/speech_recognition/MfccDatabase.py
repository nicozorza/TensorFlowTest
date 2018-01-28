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

    def sampleTrim(self, size):
        if size > self.n_frames:
            print("Size too big")

        center = int(float(self.n_frames)/2)
        return Mfcc(
            mfcc=self.mfcc[center-int(size/2):center+int(size/2)],
            label=self.label
        )

    def sampleCompleteZeros(self, size):
        if size < self.n_frames:
            print("Wrong size")
            size = self.n_frames

        aux = np.concatenate((self.mfcc, np.zeros(shape=(size-self.n_frames, self.n_mfcc))))
        return Mfcc(
            mfcc=aux,
            label=self.label
        )

    def getData(self):
        return self.mfcc

    def getLabel(self):
        return self.label

    def __str__(self):
        return 'Label: '+str(self.label)+'\n' + 'Data: '+str(self.mfcc)


class MfccDatabase(Mfcc):
    def __init__(self, mfccDatabase=None, batch_size=100):
        if mfccDatabase is None:
            self.mfccDatabase = []
        else:
            self.mfccDatabase = mfccDatabase
        self.batch_size = batch_size
        self.length = len(self.mfccDatabase)

        self.batch_count = 0
        self.batch_plan = None

    def append(self, mfcc, label):
        self.mfccDatabase.append(Mfcc(mfcc, label))
        self.length = len(self.mfccDatabase)

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

    def sampleCompleteZeros(self, size=None):
        if size is None:
            aux_size = self.mfccDatabase[0].n_frames
            for i in range(self.length):
                aux2 = self.mfccDatabase[i].n_frames
                if aux2 >= aux_size:
                    aux_size = aux2
            size = aux_size

        aux = MfccDatabase()
        for i in range(self.length):
            completed = self.mfccDatabase[i].sampleCompleteZeros(size)
            aux.append(completed.mfcc, completed.label)
        return aux

    def trainTestSet(self, factor):
        n_train = int(factor * self.length)
        aux_data = self.mfccDatabase
        random.shuffle(aux_data)
        train_set = MfccDatabase(aux_data[:n_train])
        test_set = MfccDatabase(aux_data[n_train:])
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
            data[_] = np.hstack(self.mfccDatabase[_].getData())
        return data

    def getLabels(self):
        labels = np.ndarray(
            shape=[self.length, 10]
        )
        for i in range(len(self.mfccDatabase)):
            aux = np.zeros(10)
            aux[self.mfccDatabase[i].getLabel()] = 1
            labels[i] = aux
        return labels

    def create_batch_plan(self):
        self.batch_plan = self.mfccDatabase
        random.shuffle(self.batch_plan)
        self.batch_count = 0

    def next_batch(self):
        if self.batch_count == 0:
            self.create_batch_plan()

        start_index = self.batch_size * self.batch_count
        end_index = start_index + self.batch_size
        self.batch_count += 1
        if end_index >= len(self.batch_plan):
            end_index = len(self.batch_plan)
            start_index = end_index - self.batch_size
            self.batch_count = 0

        data = MfccDatabase(self.batch_plan[start_index:end_index], self.batch_size)

        return data.getData(), data.getLabels()    #TODO CHEACKEAR ESTO

    def getMfccFromIndex(self, index):
        if index > self.length:
            return None
        return self.mfccDatabase[index]

    def getMfccFromRange(self, start_index, end_index):
        if end_index > self.length or start_index > end_index:
            return None
        return MfccDatabase(self.mfccDatabase[start_index:end_index])

    def __len__(self):
        return self.length

    def __str__(self):
        aux = ""
        for i in range(len(self.mfccDatabase)):
            aux += str(self.mfccDatabase[i]) + '\n'
        return aux