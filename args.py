import argparse

parser = argparse.ArgumentParser(description='Audio classification')

### DATASET PARAMS ###
parser.add_argument('--audio_listefile_folder_path', default='/Users/vincentbelz/Documents/Data/audio_classification/liste_file/esc50.csv', type=str)
parser.add_argument('--audio_folder_path', default='/Users/vincentbelz/Documents/Data/audio_classification/audio_files/', type=str)
parser.add_argument('--audio_images_folder_path', default='/Users/vincentbelz/Documents/Data/audio_classification/audio_images/', type=str)


parser.add_argument('--Xception_features_path', default='/Users/vincentbelz/Documents/Data/audio_classification/Xception_features/', type=str)
parser.add_argument('--InceptionResnet_features_path', default='/Users/vincentbelz/Documents/Data/audio_classification/InceptionResnet_features/', type=str)
parser.add_argument('--VGG_features_path', default='/Users/vincentbelz/Documents/Data/audio_classification/VGG_features/', type=str)


parser.add_argument('--Save_model_path', default='/Users/vincentbelz/Documents/save_model/audio_classification/', type=str)


