import tensorflow as tf
from src.speech_recognition.NeuralNetwork import  NeuralNetwork
from src.speech_recognition.MfccDatabase import MfccDatabase
import numpy as np
import pickle

MFCC_DIR = '/home/nicozorza/Escritorio/TensorflowTest/digits_database'
OUT_FILE = 'Database'

learning_rate = 0.003
n_classes = 10
n_epochs = 1000

# Load the database
file = open(MFCC_DIR+'/'+OUT_FILE, 'rb')
data = pickle.load(file)
file.close()
database = MfccDatabase(data)

# Get the MFCC matrix size
n_mfcc = database.getNMfcc()
n_frames = database.getNFrames()
batch_size = 10

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
    net_model=neural_net.neural_network_model
    )

#neural_net.predict(train_set.get_data()[0], train_set.get_labels()[0])
