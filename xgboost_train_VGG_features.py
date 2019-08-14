import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
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

# CV model
model = XGBClassifier(objective='multi:softmax',num_classes=50)
kfold = KFold(n_splits=5, random_state=7)
start_time = time.time()
results = cross_val_score(model, feat_input, y_label_sound.ravel(), cv=kfold)
end_time=time.time()
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print('result cross val',results)

print("Training time is %s seconds" % (end_time - start_time))

results = pd.DataFrame(results)
results.to_csv('/Users/vincentbelz/Documents/Data/xgb-results-02.csv', index=False)