from scipy.io import wavfile
import scipy
import numpy as np
import os
import librosa
from librosa import display
from python_speech_features import mfcc


def create_mfcc(fn):
    x, fs = librosa.load(fn)
    mfcc_features = abs(librosa.feature.mfcc(x, sr=fs))
    (rate , sig ) = scipy.io.wavfile.read(fn);
    mfcc_feat = mfcc(sig, rate, nfft = 1103)
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".mfcc"
    np.save(data_fn, mfcc_feat)


#create_fft('C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono/hiphop.00091.wav')

rootdir = ('C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono')

for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		path1 = os.path.join(subdir, file)
		create_mfcc(path1)
		