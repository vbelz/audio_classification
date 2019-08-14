import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
from args import parser
import os
import time


args = parser.parse_args()

path_save_model= args.Save_model_path

feat_input=np.load("/Users/vincentbelz/Documents/Data/audio_classification/VGG_features/VGG_feat.npy")
y_label_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/y_label_sound.npy")
y_label_hot_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/y_label_hot_sound.npy")

print('feat input shape',feat_input.shape)
print('label hot shape',y_label_hot_sound.shape)
print('label shape',y_label_sound.shape)
print('label sound values',y_label_sound)

#input_dim=8192

# grid search
model = XGBClassifier(objective='multi:softmax',num_classes=50)
n_estimators = [50, 100]
max_depth = [2, 5]
print(max_depth)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="accuracy", n_jobs=-1, cv=kfold, verbose=1)
start_time = time.time()
grid_result = grid_search.fit(feat_input, y_label_sound.ravel())
end_time = time.time()
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# Here we go

print("Training time is %s seconds" % (end_time - start_time))

results = pd.DataFrame(grid_result.cv_results_)
results.to_csv('/Users/vincentbelz/Documents/Data/xgb-random-grid-search-results-01.csv', index=False)