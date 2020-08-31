import glob
from scipy.io import wavfile
from scipy import fftpack
import numpy as np
import scipy
import librosa
import pitch
import pandas as pd
from sklearn.utils import shuffle

cols = ['aid', 'meanfreq', 'median', 'std', 'iqr', 'skew', 'melspectogram',
       'mfcc', 'spaectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
       'spectral_rolloff', 'pitch', 'Gender']

files_bdl = glob.glob('bdl/*.wav')
files_clb = glob.glob('clb/*.wav')
files_rms = glob.glob('rms/*.wav')
files_slt = glob.glob('slt/*.wav')

males = files_bdl + files_rms
females = files_clb + files_slt

gender_m = 'male'
gender_f = 'female'
features = []
for f in males:
    aid = f[-20:-17] + '_' + f[-16:-4]
    rate, audio = wavfile.read(f)
    X = fftpack.fft(audio)
    xr = X.real
    mn = xr.mean()
    md = np.median(xr)
    sd = np.std(xr)
    iqr = scipy.stats.iqr(xr)
    sk = scipy.stats.skew(xr)
    y, sr = librosa.load(f)
    mels = librosa.feature.melspectrogram(y, sr=sr)[0].mean()
    mfcc = librosa.feature.mfcc(y, sr=sr)[0].mean()
    s_centroid = librosa.feature.spectral_centroid(y, sr=sr)[0].mean()
    s_bandwith = librosa.feature.spectral_bandwidth(y, sr=sr)[0].mean()
    s_contrast = librosa.feature.spectral_contrast(y, sr=sr)[0].mean()
    s_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0].mean()
    p = round(pitch.find_pitch(f),3)
    feature = [aid, mn, md, sd, iqr, sk, mels, mfcc, s_centroid, s_bandwith, s_contrast,s_rolloff,p, gender_m]
    features.append(feature)
    
for f in females:
    aid = f[-20:-17] + '_' + f[-16:-4]
    rate, audio = wavfile.read(f)
    X = fftpack.fft(audio)
    xr = X.real
    mn = xr.mean()
    md = np.median(xr)
    sd = np.std(xr)
    iqr = scipy.stats.iqr(xr)
    sk = scipy.stats.skew(xr)
    y, sr = librosa.load(f)
    mels = librosa.feature.melspectrogram(y, sr=sr)[0].mean()
    mfcc = librosa.feature.mfcc(y, sr=sr)[0].mean()
    s_centroid = librosa.feature.spectral_centroid(y, sr=sr)[0].mean()
    s_bandwith = librosa.feature.spectral_bandwidth(y, sr=sr)[0].mean()
    s_contrast = librosa.feature.spectral_contrast(y, sr=sr)[0].mean()
    s_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0].mean()
    p = round(pitch.find_pitch(f),3)
    feature = [aid, mn, md, sd, iqr, sk, mels, mfcc, s_centroid, s_bandwith, s_contrast,s_rolloff,p, gender_f]
    features.append(feature)


data = pd.DataFrame(features, columns = cols)
data = shuffle(data)
data.to_csv('all_features.csv', index = False)
