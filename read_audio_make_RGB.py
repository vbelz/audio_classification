import numpy as np
import pandas as pd
import librosa
import os
import pywt
import cv2 as cvlib
from args import parser
import matplotlib.pyplot as plt

args = parser.parse_args()

#normalize each colour image between -1 and 1
def normalize_data_all_gather(vect_in, out_min, out_max, percent_acceptation=80,
                              not_clip_until_acceptation_time_factor=1.5):
    # nb_dim = len(vect_in.shape)
    percent_val = np.percentile(abs(vect_in).reshape((vect_in.shape[0], vect_in.shape[1] * vect_in.shape[2])),
                                percent_acceptation, axis=1)
    percent_val_matrix = not_clip_until_acceptation_time_factor * np.repeat(percent_val,
                                                                            vect_in.shape[1] * vect_in.shape[2],
                                                                            axis=0).reshape(
        (vect_in.shape[0], vect_in.shape[1], vect_in.shape[2]))
    matrix_clip = np.maximum(np.minimum(vect_in, percent_val_matrix), -percent_val_matrix)
    return np.divide(matrix_clip, percent_val_matrix) * ((out_max - out_min) / 2) + (out_max + out_min) / 2


# Read data from file 'filename.csv'
path_liste_file= args.audio_listefile_folder_path
# Control delimiters, rows, column names with read_csv (see later)
#data = pd.read_csv("/Users/vincentbelz/Documents/Data/audio_classification/liste_file/esc50.csv")
data = pd.read_csv(path_liste_file)
# Preview the first 5 lines of the loaded data
print(data.head())

print(data['filename'].iloc[250])
print(data['target'].iloc[250])
print(data['category'].iloc[250])

RGB_sound = np.zeros((2000, 299, 299, 3))
y_label_sound=np.zeros((2000,1))
y_label_hot_sound=np.zeros((2000,50))

n_fft = 1024 # frame length
hop_length = 512


path= args.audio_folder_path
path_save= args.audio_images_folder_path
#path = '/Users/vincentbelz/Documents/Data/audio_classification/audio_files/'

for i in range(2000) :

    print('loop nb',i)
    print('file name',data['filename'].iloc[i])
    print('target nb', data['target'].iloc[i])
    print('target category', data['category'].iloc[i])
    y_label_sound[i,:]=data['target'].iloc[i]
    y_label_hot_sound[i, data['target'].iloc[i]] = 1
    namefile = data['filename'].iloc[i]

    print(namefile)

    filepath = os.path.join(path, namefile)

# open the audio file
    clipnoise, sample_rate = librosa.load(filepath, sr=22500)

#print(clipnoise.shape)


    scales = np.arange(1, 128)
    waveletname = 'morl'

    coeffnoise, freqnoise = pywt.cwt(clipnoise, scales, waveletname)
#print('coeff noise',coeffnoise.shape)
#print('freq noise',freqnoise.shape)

    scalogramimg=cvlib.resize(coeffnoise, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
#print(scalogramimg.shape)

    stft = librosa.stft(clipnoise, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude, stft_phase = librosa.magphase(stft)
    stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    # print(stft_magnitude_db.shape)


    spectrogramimg=cvlib.resize(stft_magnitude_db, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
#print(spectrogramimg.shape)

# mfcc
    mfcc = librosa.feature.mfcc(y=clipnoise, sr=sample_rate, n_mfcc=200)
#print(mfcc.shape)

    mfccimg=cvlib.resize(mfcc, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)
#print(mfccimg.shape)


    RGB_sound[i, :, :, 0] = spectrogramimg
    RGB_sound[i, :, :, 1] = scalogramimg
    RGB_sound[i, :, :, 2] = mfccimg



    print('finish loop nb', i)

#print(abs(RGB_sound[0,:,:,0]).max())
#print(abs(RGB_sound[0,:,:,1]).max())
#print(abs(RGB_sound[0,:,:,2]).max())

RGB_sound[:, :, :, 0] = normalize_data_all_gather(RGB_sound[:, :, :, 0], -1, 1, 95, 2)
RGB_sound[:, :, :, 1] = normalize_data_all_gather(RGB_sound[:, :, :, 1], -1, 1, 95, 2)
RGB_sound[:, :, :, 2] = normalize_data_all_gather(RGB_sound[:, :, :, 2], -1, 1, 95, 2)



path_save_rgb= os.path.join(path_save, 'RGB_sound')
path_save_ylabel= os.path.join(path_save, 'y_label_sound')
path_save_ylabel_hot= os.path.join(path_save, 'y_label_hot_sound')

np.save(path_save_rgb,RGB_sound)
np.save(path_save_ylabel,y_label_sound)
np.save(path_save_ylabel_hot,y_label_hot_sound)

print('saved to disk')
