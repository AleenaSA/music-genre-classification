import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

GENRE_DIR ='C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono'
def read_fft(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        #print (genre_dir)
        file_list = glob.glob(genre_dir)
        #print (file_list)
        for fn in file_list:
            fft_features = np.load(fn)

            X.append(fft_features[:1000])
            y.append(label)

    
   #print (np.array(X))
    #print (np.array(y))
    return np.array(X), np.array(y)

genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]

X, y = read_fft(genre_list, 'C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono')

xTrain, xTest, yTrain, yTest = train_test_split (X, y, test_size=0.2, random_state=0)


print ("Y train : ")
print (yTrain)
print ("Y test : ")
print (yTest)