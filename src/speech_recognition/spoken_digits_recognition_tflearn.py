from src.speech_recognition.TflearnNeuralNetwork import TflearnNeuralNetwork
from src.speech_recognition.MfccDatabase import MfccDatabase
import pickle

MFCC_DIR = '/home/nicozorza/Escritorio/TensorFlowTest/digits_database'
OUT_FILE = 'Database'
OUT_MODEL_DIR = 'SavedModels/TFLearn'
OUT_MODEL = 'NetworkModel'

# Train = 0
# Validate = 1
# Train and validate = 2
train_flag = 0

learning_rate = 0.0005
n_classes = 10
n_epochs = 100

# Load the database
file = open(MFCC_DIR+'/'+OUT_FILE, 'rb')
data = pickle.load(file)
file.close()
database = MfccDatabase(data)

# Get the MFCC matrix size
n_mfcc = database.getNMfcc()
n_frames = database.getNFrames()
batch_size = 50

# Define the neural network structure
neural_net = TflearnNeuralNetwork(
    learning_rate=learning_rate,
    n_classes=n_classes,
    batch_size=batch_size,
    n_epochs=n_epochs,
    n_mfcc=n_mfcc,
    n_frames=n_frames
    )

# Define network model
neural_net.conv_neural_network_model()

# Only train
if train_flag == 0:
    # Separate train and test samples
    train_set, test_set = database.trainTestSet(0.9)

    # Save train and test sets
    file = open(MFCC_DIR + '/' + OUT_MODEL_DIR + '/' + 'train_set', 'wb')
    # Trim the samples to a fixed length
    pickle.dump(train_set.print(), file)
    file.close()
    file = open(MFCC_DIR + '/' + OUT_MODEL_DIR + '/' + 'test_set', 'wb')
    # Trim the samples to a fixed length
    pickle.dump(test_set.print(), file)
    file.close()

    # Train the network
    neural_net.train_neural_network(train_set=train_set)
    neural_net.save_model(MFCC_DIR+'/'+OUT_MODEL_DIR+'/'+OUT_MODEL)

# Only validate
if train_flag == 1:
    # Load the database
    file = open(MFCC_DIR + '/' + OUT_MODEL_DIR + '/' + 'train_set', 'rb')
    data = pickle.load(file)
    file.close()
    train_set = MfccDatabase(data)
    file = open(MFCC_DIR + '/' + OUT_MODEL_DIR + '/' + 'test_set', 'rb')
    data = pickle.load(file)
    file.close()
    test_set = MfccDatabase(data)

    # Load network model
    neural_net.load_model(MFCC_DIR+'/'+OUT_MODEL_DIR+'/'+OUT_MODEL)

    accuracy = neural_net.validate(test_set)
    print('Accuracy: ' + str(accuracy))

# Train and validate
if train_flag == 2:
    # Train and validate the network
    neural_net.train_validate_neural_network(database, 0.1)
    neural_net.save_model(MFCC_DIR + '/' + OUT_MODEL_DIR + '/' + OUT_MODEL)