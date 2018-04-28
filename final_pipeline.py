import numpy as np
import pandas as pd
#import fma
import os
import librosa
import scipy
import tensorflow as tf
from tqdm import tqdm
import multiprocessing
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

import keras
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, Embedding, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Bidirectional, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

from sklearn.utils import shuffle
from sklearn.preprocessing import normalize

import argparse

parser = argparse.ArgumentParser(description='predictions')
parser.add_argument('--test_directory', dest='test_directory', action='store', required=False)
parser.add_argument('--output_path', dest='output_path', action='store', required=False)
args = parser.parse_args()

TEST_DIRECTORY = args.test_directory
OUTPUT_PATH = args.output_path

# Parameters

feat_type = 'qstft3'

np.random.seed(42)
nb_workers = 48

num_classes = 16
rows = 128

if feat_type == 'mfcc3':
    cols = 40
elif feat_type == 'stft3':
    cols = 513
elif feat_type == 'qstft3':
    cols = 513
elif feat_type == 'other3':
    cols = 5

batch_size = 128
epochs = 500
# model_name = 'rescnn2l'+feat_type+'_b'+str(batch_size)
model_name = 'best_rescnnqstft3_b128.h5'


CLASSES = np.array(['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
               'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
               'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken'])

def stft3(file_path):
    res = []
    for i in np.arange(0.0, 28.5, 1.5):
        y, _ = librosa.load(file_path, 
                            offset = i,
                            duration = 3.0)
        if y.shape[0] < 66150:
            y = np.pad(y, (0, 66150 - y.shape[0]), 'constant')
        res.append(np.abs(librosa.stft(y, n_fft=1024, window=scipy.signal.hanning, hop_length=512))[:,:128])
    res = np.array(res)
    return res, file_path

def stft3_quick(file_path):
    res = []
    Y, _ = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
    for i in np.arange(0.0, 9.5, 0.5):
        y = Y[int(66150*i):min(len(Y),int(66150*(i+1)))]
        if y.shape[0] < 66150:
            y = np.pad(y, (0, 66150 - y.shape[0]), 'constant')
        res.append(np.abs(librosa.stft(y, n_fft=1024, window=scipy.signal.hanning, hop_length=512))[:,:128])
    res = np.array(res)
    return res, file_path

def mfcc3(file_path):
    res = []
    for i in np.arange(0.0, 28.5, 1.5):
        y, _ = librosa.load(file_path, 
                            offset = i,
                            duration = 3.0,
                            res_type='kaiser_fast')
        if y.shape[0] < 66150:
            y = np.pad(y, (0, 66150 - y.shape[0]), 'constant')
        res.append(librosa.feature.mfcc3(y=y, n_mfcc=cols)[:,:128])
    res = np.array(res)
    return res, file_path

def other3(file_path):

    res = []
    for i in np.arange(0.0, 28.5, 1.5):
        y, _ = librosa.load(file_path, 
                            offset = i,
                            duration = 3.0)
        if y.shape[0] < 66150:
            y = np.pad(y, (0, 66150 - y.shape[0]), 'constant')

        ft = []
        ft1 = ft1 = librosa.feature.rmse(y)[:,:128].T
        for k in range(128):
            ft.append(ft1[k])
        ft1 = librosa.feature.spectral_centroid(y)[:,:128].T
        for k in range(128):
            ft[k] = np.append(ft[k], ft1[k])
        ft1 = librosa.feature.spectral_bandwidth(y)[:,:128].T
        for k in range(128):
            ft[k] = np.append(ft[k], ft1[k])
        ft1 = librosa.feature.spectral_rolloff(y)[:,:128].T
        for k in range(128):
            ft[k] = np.append(ft[k], ft1[k])
        ft1 = librosa.feature.zero_crossing_rate(y)[:,:128].T
        for k in range(128):
            ft[k] = np.append(ft[k], ft1[k])

        ft = np.array(ft)

        res.append(ft.T)

    res = np.array(res)
    return res, file_path

# def train_model():
#     print('-'*130)
#     print ('Features extracting: train')
#     print('-'*130)

#     flty = fma.FILES_TRAIN_FAULTY
#     names = os.listdir('data/fma_medium/')
#     names.remove('README.txt')
#     names.remove('checksums')

#     files = []
#     for name in names:
#         i_names = os.listdir('data/fma_medium/{}/'.format(name))
#         for n in i_names:
#             if int(n[:6]) in flty:
#                 continue
#             files.append('data/fma_medium/{}/{}'.format(name, n))

#     x_train = []
#     y_train = []
#     names = []

#     # files = files[:100]

#     pool = multiprocessing.Pool(nb_workers)

#     if feat_type == 'mfcc3':
#         it = pool.imap_unordered(mfcc3, files)
#     elif feat_type == 'stft3':
#         it = pool.imap_unordered(stft3, files)
#     elif feat_type == 'qstft3':
#         it = pool.imap_unordered(stft3_quick, files)
#     elif feat_type == 'other3':
#         it = pool.imap_unordered(other3, files)
#     for data, file_path in tqdm(it, total=len(files)):
#         for i in range(19):
#             x_train.append(data[i].T)
#             names.append(file_path[20:26])

#     pool.close()

#     labels = pd.read_csv('data/train_labels.csv', index_col=0, squeeze=True)
#     labels = labels.drop(fma.FILES_TRAIN_FAULTY)

#     class_map = {}
#     cnt = 0
#     for cl in CLASSES:
#         class_map[cl] = cnt
#         cnt += 1

#     for name in names:
#         y_train.append(class_map[labels[int(name)]])

#     x_train = np.asarray(x_train)
#     y_train = np.asarray(y_train)

#     print('-'*130)
#     print ('Model train')
#     print('-'*130)

#     input_shape = (rows, cols, 1)

#     inputs = Input(shape=input_shape)

#     x = BatchNormalization()(inputs)
#     x = Conv2D(256, kernel_size=(4, cols), activation='relu', input_shape=input_shape)(x)
#     shortcut = x
#     x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)
#     x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)

#     x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)
#     x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)

#     x1 = AveragePooling2D(pool_size=(125, 1))(keras.layers.concatenate([x, shortcut]))
#     x2 = MaxPooling2D(pool_size=(125, 1))(keras.layers.concatenate([x, shortcut]))

#     # z = MaxPooling2D(pool_size=(2, 1))(inputs)
#     # z = Lambda(lambda y: K.squeeze(y, 3))(z)
#     # # z = Embedding(input_dim=200000, output_dim=128, input_length=100)(z)
#     # z = Bidirectional(LSTM(256, return_sequences=False))(z)
#     # z = Lambda(lambda y: K.reshape(y, (-1,1,1,512)))(z)

#     x = Dropout(0.2)(keras.layers.concatenate([x1, x2]))
#     x = Flatten()(x)
#     x = Dense(480, activation='relu')(x)
#     x = Dropout(0.2)(x)
#     x = Dense(240, activation='relu')(x)
#     x = Dropout(0.2)(x)

#     pred = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=inputs, outputs=pred)

#     x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
#     y_train = to_categorical(y_train, num_classes)

#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adadelta(),
#                   metrics=['acc'])

#     checkpoint = ModelCheckpoint('best_'+model_name+'.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
#     early_stop = EarlyStopping(monitor='acc', patience=5, mode='max') 
#     callbacks_list = [checkpoint, early_stop]

#     model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               shuffle=True,
#               callbacks=callbacks_list)


#     # model.save(model_name+'.h5')

#     del x_train

def predict_model():
    bs = 5000
    FILES = []
    names = []
    y_pred = np.empty((0,16))

    #i_test_names = os.listdir('data/crowdai_fma_test/')
    i_test_names = os.listdir(TEST_DIRECTORY)
    for name in i_test_names:
        #FILES.append('data/crowdai_fma_test/{}'.format(name))
        FILES.append(TEST_DIRECTORY + '{}'.format(name))

    FILES = np.array(FILES)
    #model = load_model('best_'+model_name+'.h5') 
    model = load_model(model_name)

    pool = multiprocessing.Pool(nb_workers)

    for ind in np.arange(0, len(FILES), bs):

        print('-'*130)
        print ('Features extracting: test batch [', ind, '-', min(ind+bs, len(FILES)), ']')
        print('-'*130)

        files = FILES[ind:min(ind+bs, len(FILES))]
        x_test = []        

        if feat_type == 'mfcc3':
            it = pool.imap_unordered(mfcc3, files)
        elif feat_type == 'stft3':
            it = pool.imap_unordered(stft3, files)
        elif feat_type == 'qstft3':
            it = pool.imap_unordered(stft3_quick, files)
        elif feat_type == 'other3':
            it = pool.imap_unordered(other3, files)
        for data, file_path in tqdm(it, total=len(files)):
            names.append(file_path)
            for i in range(19):
                x_test.append(data[i].T)

        print('-'*130)
        print ('Predicting: test batch [', ind, '-', min(ind+bs, len(FILES)), ']')
        print('-'*130)

        x_test = np.array(x_test)
        x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
        y_pred = np.vstack([y_pred, model.predict(x_test)])

        # del x_test

    pool.close()

    y_pred_final = []
    for i in range(y_pred.shape[0]//19):
        res = []
        for j in range(19):
            res.append(y_pred[i*19+j])
        res = np.array(res)
        y_pred_final.append(np.mean(res, axis=0))

    y_pred_final = np.array(y_pred_final)
    names = np.array(names)

    res = sorted(zip(names, y_pred_final))
    names, y_test = zip(*res)

    #names_new = np.array([s[22:58] for s in names])
    names_new = np.array([os.path.basename(s) for s in names])
    y_test = np.array(y_test)

    print (y_test.shape)

    submission = pd.DataFrame(y_test, names_new, CLASSES)
    submission.index.name = 'file_id'

    #submission.to_csv('submission_'+model_name+'.csv', header=True)
    #submission.to_csv('/tmp/output.csv', header=True)
    submission.to_csv(OUTPUT_PATH, header=True)
    print ('submission_'+model_name+'.csv')


# train_model()
predict_model()
