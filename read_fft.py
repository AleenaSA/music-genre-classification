import numpy as np
import os
import glob

GENRE_DIR ='C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono'
def read_fft(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        print (genre_dir)
        file_list = glob.glob(genre_dir)
        #print (file_list)
        for fn in file_list:
            fft_features = np.load(fn)

            X.append(fft_features[:1000])
            y.append(label)

    
    print (np.array(X))
    print (np.array(y))
    return np.array(X), np.array(y)

genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]

read_fft(genre_list, 'C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono')