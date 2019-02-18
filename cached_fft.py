from scipy.io import wavfile
import scipy
import numpy as np
import os

def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    fft_features = abs(scipy.fft(X)[:1000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)


#create_fft('C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono/hiphop.00091.wav')

rootdir = ('C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/temp_genres')

for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		path1 = os.path.join(subdir, file)
		create_fft(path1)