from src.speech_recognition.TflearnNeuralNetwork import TflearnNeuralNetwork
from src.speech_recognition.MfccDatabase import MfccDatabase
import pickle

MFCC_DIR = '/home/nicozorza/Escritorio/TensorFlowTest/digits_database'
OUT_FILE = 'Database'
OUT_MODEL_DIR = 'SavedModels'
OUT_MODEL = 'TflearnNetworkModel'

train_flag = True

learning_rate = 0.0005
n_classes = 10
n_epochs = 200

# Load the database
file = open(MFCC_DIR+'/'+OUT_FILE, 'rb')
data = pickle.load(file)
file.close()
database = MfccDatabase(data)

# Get the MFCC matrix size
n_mfcc = database.getNMfcc()
n_frames = database.getNFrames()
batch_size = 20

# Separate train and test samples
train_set, test_set = database.trainTestSet(0.9)

# Define the neural network structure
neural_net = TflearnNeuralNetwork(
    learning_rate=learning_rate,
    n_classes=n_classes,
    batch_size=batch_size,
    n_epochs=n_epochs,
    n_mfcc=n_mfcc,
    n_frames=n_frames
    )

model = neural_net.conv_neural_network_model()

if train_flag:
    # Train the network
    neural_net.train_neural_network(
        train_set=train_set,
        net_model=model,
        )
    neural_net.save_model(MFCC_DIR+'/'+OUT_MODEL_DIR+'/'+OUT_MODEL, model)

    print("Train size: " + str(train_set.length))
    print("Test size: " + str(test_set.length))

neural_net.load_model(MFCC_DIR+'/'+OUT_MODEL_DIR+'/'+OUT_MODEL, model)
test_data = test_set.getMfccFromIndex(0)

predicted = neural_net.predict(test_data, model)
print('Label: ' + str(test_data.label) + ' --- ' + 'Predicted: ' + str(predicted))

accuracy = neural_net.validate(test_set, model)
print('Accuracy: ' + str(accuracy))
