import numpy as np
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from args import parser
import os
import time
from sklearn.model_selection import train_test_split, StratifiedKFold

args = parser.parse_args()

path_save_model= args.Save_model_path

print("simple")
print("Keras version:",keras.__version__)
print("Tensorflow version:",tensorflow.VERSION)

feat_input=np.load("/Users/vincentbelz/Documents/Data/audio_classification/Xception_features/Xception_feat.npy")
y_label_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/y_label_sound.npy")
y_label_hot_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/y_label_hot_sound.npy")

print('feat input shape',feat_input.shape)
print('label shape',y_label_hot_sound.shape)

modeltop = Sequential()

# in the first layer, you must specify the expected input data shape:
modeltop.add(Dense(1024, activation='sigmoid', input_dim=73728))
modeltop.add(Dense(2048, activation='relu'))
modeltop.add(Dropout(0.2))
modeltop.add(Dense(256, activation='relu'))
modeltop.add(Dropout(0.1))
modeltop.add(Dense(64, activation='relu'))
modeltop.add(Dense(50, activation='softmax'))

# Set Optimizer
opt = Adam(lr=0.001, decay=1e-6)

modeltop.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train=feat_input
y_train=y_label_hot_sound
y_train_for_kfold=y_label_sound


folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(x_train, y_train_for_kfold))

print('folds list', folds)

start_time = time.time()

result=[]

for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ', j)
    x_train_cv = x_train[train_idx]
    y_train_cv = y_train[train_idx]
    x_valid_cv = x_train[val_idx]
    y_valid_cv = y_train[val_idx]
    modeltop.fit(x_train_cv, y_train_cv, epochs=50, batch_size=32, shuffle=True, verbose=1, validation_data=(x_valid_cv, y_valid_cv))
    print(modeltop.evaluate(x_valid_cv, y_valid_cv))
    result.append(modeltop.evaluate(x_valid_cv, y_valid_cv))

print('final result',result)

end_time = time.time()

print("Training time is %s seconds" % (end_time - start_time))