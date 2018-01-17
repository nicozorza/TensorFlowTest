
import librosa
import speechpy
import pickle
import os
from src.speech_recognition.MfccDatabase import MfccDatabase
import matplotlib.pyplot as plt

import numpy as np


MFCC_DIR = '/home/nicozorza/Escritorio/TensorflowTest/digits_database'
WAV_DIR = MFCC_DIR+'/wav'
OUT_FILE = 'Database'

# Get the names of the directories in the wav database
NUMBERS_DIR_LIST = os.listdir(WAV_DIR)

n_mfcc = 15                 # Number of MFCC coefficients
preemphasis_coeff = 0.98
frame_length = 0.02         # Length of the frame window
frame_stride = 0.01         # Slide of the window
fft_points = 1024
num_filters = 40            # Number of filters in the filterbank

database = MfccDatabase()

figure = 0
show_figures = False

wav_counter = 0
for dirs in range(len(NUMBERS_DIR_LIST)):
    # Get the name of the directory
    dir_name = NUMBERS_DIR_LIST[dirs]
    # Get the names of each wav file in the directory
    wav_list_path = os.listdir(WAV_DIR+'/'+dir_name)

    for wavs in range(len(wav_list_path)):
        wav_counter += 1
        wav_name = wav_list_path[wavs].split('.')[0]    # Get the wav name without extension

        # Read the wav file
        signal, fs = librosa.load(WAV_DIR+'/'+NUMBERS_DIR_LIST[dirs]+'/'+wav_list_path[wavs])
        # Apply a pre-emphasis filter
        signal_preemphasized = speechpy.processing.preemphasis(signal=signal, cof=preemphasis_coeff)
        # Get the MFCCs coefficients. The size of the matrix is n_mfcc x T, so the dimensions
        # are not the same for every sample
        mfcc = speechpy.feature.mfcc(signal,
                                     sampling_frequency=fs,
                                     frame_length=frame_length,
                                     frame_stride=frame_stride,
                                     num_filters=num_filters,
                                     fft_length=fft_points,
                                     low_frequency=0,
                                     high_frequency=None,
                                     num_cepstral=n_mfcc)
        # Get the normalization factor
        factor = np.amax([np.amax(mfcc), np.abs(np.amin(mfcc))])
        mfcc = mfcc / factor

        # Add the new data to the database
        database.append(
            label=int(dir_name),    # The directory name is the same as the label
            mfcc=mfcc
        )
        if wavs == 0 and show_figures:
            plt.figure(num=figure, figsize=(2, 2))
            figure = figure + 1
            heatmap = plt.pcolor(mfcc)
            plt.title(wav_name)
            plt.draw()

        print('Wav', wav_counter, 'completed out of', len(wav_list_path)*len(NUMBERS_DIR_LIST), 'Label: ', dir_name)
if show_figures:
    plt.show()

# Save the database into a file
file = open(MFCC_DIR + '/' + OUT_FILE, 'wb')
# Trim the samples to a fixed length
pickle.dump(database.sampleCompleteZeros().print(), file)
file.close()

print("Database generated")
print("Number of elements in database: " + str(database.length))