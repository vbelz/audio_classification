import numpy as np
import pandas as pd
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
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

feat_input=np.load("/Users/vincentbelz/Documents/Data/audio_classification/VGG_features/VGG_feat.npy")
y_label_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/y_label_sound.npy")
y_label_hot_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/y_label_hot_sound.npy")

print('feat input shape',feat_input.shape)
print('label shape',y_label_hot_sound.shape)


def create_model(neuronslay1=1024, neuronslay2=512, neuronslay3=256, neuronslay4=128, dropout_ratelay1=0.0, dropout_ratelay2=0.0, dropout_ratelay3=0.0, activation='sigmoid'):
	# create model
	model = Sequential()
	model.add(Dense(neuronslay1, input_dim=8192, activation=activation))
	model.add(Dropout(dropout_ratelay1))
	model.add(Dense(neuronslay2, activation='relu'))
	model.add(Dropout(dropout_ratelay2))
	model.add(Dense(neuronslay3, activation='relu'))
	model.add(Dropout(dropout_ratelay3))
	model.add(Dense(neuronslay4, activation='relu'))
	model.add(Dense(50, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# define the grid search parameters
activation = ['relu', 'sigmoid']
neuronslay1 = [1024]
neuronslay2 = [512]
neuronslay3 = [256]
neuronslay4 = [128]
dropout_ratelay1 = [0.2]
dropout_ratelay2 = [0.2]
dropout_ratelay3 = [0.2]

x_train=feat_input
y_train=y_label_hot_sound
y_train_for_kfold=y_label_sound


# create model
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=1)
# define the grid search parameters
param_grid = dict(neuronslay1=neuronslay1, neuronslay2=neuronslay2, neuronslay3=neuronslay3, neuronslay4=neuronslay4, dropout_ratelay1= dropout_ratelay1, dropout_ratelay2=dropout_ratelay2, dropout_ratelay3=dropout_ratelay3, activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

results = pd.DataFrame(grid_result.cv_results_)
results.to_csv('/Users/vincentbelz/Documents/Data/NN-VGG-grid-search-results-01.csv', index=False)