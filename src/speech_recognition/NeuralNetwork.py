import tensorflow as tf
from src.speech_recognition.Database import Database
import numpy as np

class NeuralNetwork:

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
        self.x = tf.placeholder(dtype='float', shape=[None, self.n_frames * self.n_mfcc], name='input')
        self.y = tf.placeholder(dtype='float', shape=[None, self.n_classes], name='output')
        self.graph_def = None

    def neural_network_model(self):

        n_nodes_hl1 = 300
        n_nodes_hl2 = 100
        n_nodes_hl3 = 50

        hidden_l1 = {
            'weights': tf.Variable(tf.random_normal([self.n_mfcc*self.n_frames, n_nodes_hl1])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
        }
        # hidden_l2 = {
        #     'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        #     'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
        # }
        hidden_l3 = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl3])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
        }
        output_l4 = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl3, self.n_classes])),
            'biases': tf.Variable(tf.random_normal([self.n_classes]))
        }

        l1 = tf.add(tf.matmul(self.x, hidden_l1['weights']), hidden_l1['biases'])
        l1 = tf.nn.sigmoid(l1)

        # l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
        # l2 = tf.nn.sigmoid(l2)

        l3 = tf.add(tf.matmul(l1, hidden_l3['weights']), hidden_l3['biases'])
        l3 = tf.nn.sigmoid(l3)

        output = tf.add(tf.matmul(l3, output_l4['weights']), output_l4['biases'])
        return output

    def conv_neural_network_model(self):

        input_layer = tf.reshape(self.x, [-1, self.n_frames, self.n_mfcc, 1])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=10,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=10,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 200])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

        logits = tf.layers.dense(inputs=dropout, units=10)

        return logits

    def train_neural_network(self, train_set, test_set, net_model):
        prediction = net_model()

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epochs):
                epoch_loss = 0

                for _ in range(int(train_set.length / self.batch_size)):
                    epoch_x, epoch_y = train_set.next_batch(self.batch_size)

                    _, c = sess.run(
                        [optimizer, cost], feed_dict={
                            self.x: epoch_x,
                            self.y: epoch_y
                        }
                    )
                    epoch_loss += c
                print('Epoch', epoch, 'completed out of', self.n_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(input=prediction, axis=1), tf.argmax(input=self.y, axis=1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval({
                self.x: test_set.getData(),
                self.y: test_set.getLabels()
            }))
