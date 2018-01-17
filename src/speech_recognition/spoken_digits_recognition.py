import tensorflow as tf
from src.speech_recognition.NeuralNetwork import  NeuralNetwork
from src.speech_recognition.MfccDatabase import MfccDatabase
import numpy as np
import pickle

MFCC_DIR = '/home/nicozorza/Escritorio/TensorflowTest/digits_database'
OUT_FILE = 'Database'

learning_rate = 0.01
n_classes = 10
n_epochs = 10

# Load the database
file = open(MFCC_DIR+'/'+OUT_FILE, 'rb')
data = pickle.load(file)
file.close()
database = MfccDatabase(data)

# Get the MFCC matrix size
n_mfcc = database.getNMfcc()
n_frames = database.getNFrames()
batch_size = 50

# Separate train and test samples
train_set, test_set = database.trainTestSet(0.9)

# Define the neural network structure
neural_net = NeuralNetwork(
    learning_rate=learning_rate,
    n_classes=n_classes,
    batch_size=batch_size,
    n_epochs=n_epochs,
    n_mfcc=n_mfcc,
    n_frames=n_frames
    )

# Train the network
neural_net.train_neural_network(
    train_set=train_set,
    test_set=test_set,
    net_model=neural_net.conv_neural_network_model
    )

print("Train size: " + str(train_set.length))
print("Test size: " + str(test_set.length))

#neural_net.predict(train_set.get_data()[0], train_set.get_labels()[0])

