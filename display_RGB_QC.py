# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pandas as pd
import librosa
import os
import pywt
import cv2 as cvlib
from args import parser
import matplotlib.pyplot as plt

RGB_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/RGB_sound.npy")
y_label_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/y_label_sound.npy")
y_label_hot_sound=np.load("/Users/vincentbelz/Documents/Data/audio_classification/audio_images/y_label_hot_sound.npy")

print(RGB_sound.shape)
print(y_label_sound[1199,:])
print(y_label_hot_sound[1199,:])

plt.figure()
plt.axis('off')
plt.imshow(RGB_sound[1199,:,:,0], cmap='Reds', interpolation='nearest', aspect='auto')
plt.show()

plt.figure()
plt.axis('off')
plt.imshow(RGB_sound[1199,:,:,1], cmap='Greens', interpolation='nearest', aspect='auto')
plt.show()

plt.figure()
plt.axis('off')
plt.imshow(RGB_sound[1199,:,:,2],clim=(-0.2, 0.2), cmap='Blues', interpolation='nearest', aspect='auto')
plt.show()