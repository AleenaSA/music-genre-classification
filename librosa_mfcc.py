import librosa
from librosa import display

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pylab
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

def plot_confusion_matrix(cm, genre_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues')
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.show()

GENRE_DIR ='C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_mono_mfcc'
def read_mfcc(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    MFCCs = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.mfcc.npy")
        #print (genre_dir)
        file_list = glob.glob(genre_dir)
        #print (file_list)
        for fn in file_list:
            ceps = np.load(fn)
            num_ceps = len(ceps)

            X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
            y.append(label)
  


    """min_shape = (20, 345)
    for idx, arr in enumerate(MFCCs):
    	MFCCs[idx] = arr[:, :min_shape[1]]"""

    return np.array(X), np.array(y)

genre_list = ["classical", "reggae", "blues", "hiphop", "disco", "metal"]

X, y = read_mfcc(genre_list, 'C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_mono_mfcc')


xTrain, xTest, yTrain, yTest = train_test_split (X, y, test_size=0.2, random_state=0)

xTest_temp=[]
xTrain_temp=[]

for x in xTrain:
	x = x[:13]
	xTrain_temp.append(x)
	#print(len(x))

for x in xTest:
	x = x[:13]
	xTest_temp.append(x)



#print ("Y test : ")
#print (yTrain)

'''reg = LogisticRegression()
reg.fit(xTrain_temp, yTrain)



#print (reg.coef_)
#print(reg.intercept_)
predictions = reg.predict(xTest_temp)
print (yTest)
print (predictions)
cm=confusion_matrix(yTest, predictions)
print (reg.score(xTest_temp, yTest))
print (cm)
plot_confusion_matrix(cm, genre_list, 'Name', 'Confusion Matrix')'''

#Using SVM

clf=svm.SVC(gamma='scale')
clf.fit(xTrain_temp, yTrain)
predictions=clf.predict(xTest_temp)
cm=confusion_matrix(yTest, predictions)
print (clf.score(xTest_temp, yTest))
print (cm)
#print ( list (cm))
#df_cm = pd.DataFrame(list(cm), range(6), range(6))
#plt.figure(figsize = (10,7))
#sn.heatmap(df_cm, annot=True)
plot_confusion_matrix(cm, genre_list, 'Name', 'Confusion Matrix')


