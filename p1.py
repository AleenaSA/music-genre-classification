"""import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono/country.00001.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()"""

import scipy
import matplotlib.pyplot as plt
import scipy.io.wavfile
sample_rate1, X1 = scipy.io.wavfile.read('C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono/classical.00001.wav')
print (sample_rate1, X1.shape )
plt.specgram(X1, Fs=sample_rate1, xextent=(0,30))
plt.show()

sample_rate2, X2 = scipy.io.wavfile.read('C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono/hiphop.00091.wav')
print (sample_rate2, X2.shape )
plt.specgram(X2, Fs=sample_rate2, xextent=(0,30))
plt.show()

sample_rate3, X3 = scipy.io.wavfile.read('C:/Users/Aleena/Desktop/SEM6/Machine Learning/Music genre classifications/genres_wav_mono/jazz.00051.wav')
print (sample_rate3, X3.shape )
plt.specgram(X3, Fs=sample_rate3, xextent=(0,30))
plt.show()