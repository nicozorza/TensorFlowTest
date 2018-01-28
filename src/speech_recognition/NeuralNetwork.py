import tensorflow as tf
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
        self.y = tf.placeholder(dtype='float', shape=[None, self.n_classes], name='labels')
        self.graph_def = None
        self.saver = None

    def neural_network_model(self):

        n_nodes_hl1 = 30
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
        conv1_filters = 12
        conv2_filters = 4
        pool2_flat_size = int(self.n_frames/4)*int(self.n_mfcc/4)*conv2_filters
        pool2_flat_dense_size = 100

        input_layer = tf.reshape(self.x, [-1, self.n_frames, self.n_mfcc, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=conv1_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=conv2_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, pool2_flat_size])
        dense = tf.layers.dense(inputs=pool2_flat, units=pool2_flat_dense_size, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

        logits = tf.layers.dense(inputs=dropout, units=10)

        return logits

    def train_neural_network(self, train_set, test_set, net_model, model_file_name):
        prediction = net_model()
        output = tf.argmax(input=prediction, axis=1, name='output')
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epochs):
                epoch_loss = 0

                for _ in range(int(train_set.length / self.batch_size)):
                    epoch_x, epoch_y = train_set.next_batch()

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

            self.graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])

    def predict(self, test_mfcc):
        # TODO Esto es medio desprolijo
        if not self.graph_def:
            print("No model trained")
            return

        input, output = tf.import_graph_def(self.graph_def,
                                            return_elements=[
                                                "input:0",
                                                "output:0"]
                                            )

        data = np.ndarray(
            shape=[1, test_mfcc.n_mfcc*test_mfcc.n_frames],
        )

        data[0] = np.hstack(test_mfcc.getData())
        with tf.Session() as session:
            aux = session.run([output], feed_dict={input: data})[0]
            return aux[0]

    def save_model(self, file_name):

        if not self.graph_def:
            print("No model trained")
            return

        with open(file_name, 'wb') as f:
            f.write(self.graph_def.SerializeToString())

    def load_model(self, file_name):

        with open(file_name, "rb") as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

