import numpy as np
import pandas as pd
import librosa
import os
import pywt
import cv2 as cvlib
from args import parser
import matplotlib.pyplot as plt
from tsfresh import extract_features


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

path= args.audio_folder_path
path_save= args.audio_images_folder_path
#path = '/Users/vincentbelz/Documents/Data/audio_classification/audio_files/'



namefile = data['filename'].iloc[0]

print(namefile)

for i in range(1) :

    print('loop nb',i)
    print('file name',data['filename'].iloc[i])
    print('target nb', data['target'].iloc[i])
    print('target category', data['category'].iloc[i])
    namefile = data['filename'].iloc[i]

    print(namefile)

    filepath = os.path.join(path, namefile)

# open the audio file
    clipnoise, sample_rate = librosa.load(filepath, sr=22500)
    print('clipnoise shape',clipnoise.shape)
    print('sample_rate',sample_rate)
    clipfirstsecond=clipnoise[:112500]
    time=np.arange(112500)
    filenb=np.full((112500, ), i, dtype=int)
    print('first second',clipfirstsecond.shape)
    print('time',time.shape)
    print(filenb.shape)
    clipfirstsecond=clipfirstsecond.T
    time=time.T
    filenb=filenb.T
    concat=np.zeros((112500,3))
    concat[:,0]=filenb
    concat[:,1]=time
    concat[:,2]=clipfirstsecond
    print('concat numpy',concat.shape)
    print('concat', concat)
    df = pd.DataFrame(concat, columns=['id','time','amplitude'])
    df['id'] = df['id'].astype('int64')
    print('df top',df.head())
    print('df bottom', df.tail())
    print('max',df['amplitude'].max())
    print('df types',df.dtypes)
    extracted_features = extract_features(df, column_id="id", column_sort="time")
    print(extracted_features)



