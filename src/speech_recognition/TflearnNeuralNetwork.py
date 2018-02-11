import tflearn
import numpy as np


class TflearnNeuralNetwork:

    def __init__(self,
                 learning_rate,
                 n_classes,
                 batch_size,
                 n_epochs,
                 n_mfcc,
                 n_frames):

        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_mfcc = n_mfcc
        self.n_frames = n_frames
        self.model = None

    def neural_network_model(self):
        n_nodes_hl1 = 100
        n_nodes_hl2 = 90
        # n_nodes_hl3 = 15

        net = tflearn.input_data(shape=[None, self.n_frames * self.n_mfcc])
        net = tflearn.fully_connected(net, n_nodes_hl1, activation='tanh', regularizer='L2')
        net = tflearn.fully_connected(net, n_nodes_hl2, activation='tanh', regularizer='L2')
        # net = tflearn.fully_connected(net, n_nodes_hl3, activation='tanh', regularizer='L2')
        net = tflearn.fully_connected(net, self.n_classes, activation='softmax')
        net = tflearn.regression(net,
                                 optimizer='adam',
                                 learning_rate=self.learning_rate,
                                 loss='categorical_crossentropy',
                                 name='target')

        self.model = tflearn.DNN(net, tensorboard_verbose=0)

        return self.model

    def conv_neural_network_model(self):
        conv1_filters = 10
        conv2_filters = 6
        pool2_flat_dense_size = 100

        network = tflearn.input_data(shape=[None, self.n_frames * self.n_mfcc], name='input')
        network = tflearn.reshape(network, new_shape=[-1, self.n_frames, self.n_mfcc, 1])
        network = tflearn.conv_2d(network, conv1_filters, 5, activation='relu', regularizer="L2")
        network = tflearn.max_pool_2d(network, 2)
        network = tflearn.local_response_normalization(network)
        network = tflearn.conv_2d(network, conv2_filters, 5, activation='relu', regularizer="L2")
        network = tflearn.max_pool_2d(network, 2)
        network = tflearn.local_response_normalization(network)
        network = tflearn.fully_connected(network, pool2_flat_dense_size, activation='tanh', regularizer="L2")
        #network = tflearn.dropout(network, 0.1)
        network = tflearn.fully_connected(network, self.n_classes, activation='softmax')
        network = tflearn.regression(
            network,
            optimizer='adam',
            learning_rate=self.learning_rate,
            loss='categorical_crossentropy',
            name='target')

        self.model = tflearn.DNN(network, tensorboard_verbose=0)

        return self.model

    def train_neural_network(self, train_set, net_model=None):
        if net_model is None:
            net_model = self.model
        net_model.fit(
            train_set.getData(),
            train_set.getLabels(),
            n_epoch=self.n_epochs,
            batch_size=self.batch_size,
            show_metric=True)

    def train_validate_neural_network(self, complete_set, val_factor, net_model=None):
        if net_model is None:
            net_model = self.model
        net_model.fit(
            complete_set.getData(),
            complete_set.getLabels(),
            validation_set=val_factor,
            n_epoch=self.n_epochs,
            batch_size=self.batch_size,
            show_metric=True)

    def predict(self, test_mfcc, net_model=None):
        if net_model is None:
            net_model = self.model

        data = np.ndarray(shape=[1, self.n_mfcc * self.n_frames])
        data[0] = np.hstack(test_mfcc.getData())
        label = net_model.predict_label(data)[0][0]
        return label

    def validate(self, test_set, net_model=None):
        if net_model is None:
            net_model = self.model
        counter = 0

        for i in range(test_set.length):
            test_mfcc = test_set.getMfccFromIndex(i)
            prediction = self.predict(test_mfcc, net_model)
            if prediction == test_mfcc.getLabel():
                counter += 1
                print('\033[94m' + 'Label: ' + str(test_mfcc.getLabel()) + ' --- ' + 'Predicted: ' + str(prediction) + '\033[0m')
            else:
                print('\033[91m' + 'Label: ' + str(test_mfcc.getLabel()) + ' --- ' + 'Predicted: ' + str(prediction) + '\033[0m')
        return (counter / test_set.length)*100

    def save_model(self, file_name, model=None):
        if model is None:
            model = self.model
        model.save(file_name)

    def load_model(self, file_name, model=None):
        if model is None:
            model = self.model
        model.load(file_name)
        self.model = model
        return model