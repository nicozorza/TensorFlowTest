import tensorflow as tf
# Ejemplo para el reconocimiento de dígitos utilizando una red
# neuronal convolucional

# Base de datos con los numeros 0-9 escritos a mano
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01

# Cantidad de clases. Es igual a la cantidad de salidas, ya que
# la salida de la red es un vector de 10 componentes:
# out=[1 0 ... 0 0] -> Se reconocio un 0
# out=[0 0 1 ... 0] -> Se reconocio un 2
n_classes = 10

# Subgrupo de imagenes utilizadas para entrenar. Esto permite optimizar las
# operaciones matriciales.
batch_size = 100

# Se cumple un epoch cuando se propagaron los pesos hacia la salida
# y se ejecuta backpropagation. Epoch = propagation + backpropagation (ida y vuelta)
n_epochs = 5

# Las imagenes tienen 28x28 pixeles, y son reorganizadas en un array.
n_pixels = 28*28

# Se define la estructura de los datos de entrada y las etiquetas de salida.
# De entrada se tienen batch_size=100 imagenes, que estan representadas por vectores
# de largo n_pixels=28*28. En lugar de utilizar batch_size se utiliza None, porque luego
# se contrastará contra todas las test_images, y por ende la estructura debe poder ser
# aplicada a dichas imagenes.
x = tf.placeholder(dtype='float', shape=[None, n_pixels])
# Se definen las batch_size=100 etiquetas, donde cada una de ellas es un vector
# de n_classes=10 componentes.
y = tf.placeholder('float', shape=[None, n_classes])


# Esta función determina la estructura de la red convolucional.
def neural_network_model(data):

    # Se transforma el vector de entrada en una matriz
    input_layer = tf.reshape(data, [-1, 28, 28, 1])

    # Primer capa convolucional
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Capa de max-pooling
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Segunda capa convolucional
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Capa de max-pooling
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Se convierte el feature map en un vector
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # Dense Layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # Se aplica la regularizacion de dropout
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    # Se agrupan los resultados en 10 salidas, una por clase
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits


def train_neural_network(x_train):
    prediction = neural_network_model(x_train)  # Se crea la red

    # Se deine la funcion de costo: cross-entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # Se define el optimizador. AdamOptimizer es similar al metodo de gradiente descendente estocastico.
    # Se establece tambien el learning_rate, y se indica que se desea minimizar la funcion de costo
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Hasta ahora solo se estaba definiendo la estructura.
    # Se comienza el TRAIN.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # Se inicializan las variables

        for epoch in range(n_epochs):   # Se entrena n_epochs veces
            epoch_loss = 0
            # Se utilizan todos los ejemplos de mnist, tomandolos de a porciones
            # del tamaño de batch_size
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # Se eligen batch_size muestras de la base de datos. Esta funcion retorna
                # las imagenes y sus respectivos labels
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # Se ejecuta el optimizador y se calcula el costo.
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        # Se comienza el TEST
        # Se comparan las predicciones con las etiquetas. Se busca cual de las salidas es la mayor, y
        # se establece que dicha neurona es la ganadora. Se compara entonces ese valor con la etiqueta
        # ganadora para ver si coinciden
        correct = tf.equal(tf.argmax(input=prediction, axis=1), tf.argmax(input=y, axis=1))

        # Se hace un promedio de las predicciones correctas
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # Se ejecuta el testeo. Se utilizan como argumento las imagenes de testeo, que no son las
        # mismas que las utilizadas para el entrenamiento. Se debe notar que el batch_size es diferente
        # para el testeo, ya que se estan pasando todas las imagenes de una
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


# Se entrena la red y se observan sus resultados
train_neural_network(x)