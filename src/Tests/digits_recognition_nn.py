import tensorflow as tf
# Ejemplo para el reconocimiento de dígitos utilizando una red
# neuronal convencional

# Base de datos con los numeros 0-9 escritos a mano
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01

n_nodes_hl1 = 500   # Cantidad de nodos de la capa oculta 1
n_nodes_hl2 = 500   # Cantidad de nodos de la capa oculta 2
n_nodes_hl3 = 500   # Cantidad de nodos de la capa oculta 3

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
n_epochs = 10

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


# Esta función determina la estructura de las capas ocultas y de la salida, es
# decir la estructura de la red.
def neural_network_model(data):
    # Los pesos y bias se inicializan de manera aleatoria. La salida de cada
    # capa es conectada a la entrada de la capa siguiente.
    hidden_l1 = {
        'weights': tf.Variable(tf.random_normal([n_pixels, n_nodes_hl1])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }
    hidden_l2 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }
    hidden_l3 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
    }
    output_l4 = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    # La salida de las capas se calcula como: z=W*x+b. Luego se aplica la funcion de activacion
    # deseada, y se pasa a la capa siguiente.
    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['biases'])
    l1 = tf.nn.sigmoid(l1)  # Se aplica la funcion de activacion

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases'])
    l3 = tf.nn.sigmoid(l3)

    # Se utiliza una salida lineal
    output = tf.add(tf.matmul(l3, output_l4['weights']), output_l4['biases'])
    return output


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