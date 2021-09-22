import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.io
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.applications import vgg16
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import time


def VGGModel(lr):
    input_shape = (224, 224, 3)

    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
               activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(rate=0.1),
        Dense(2048, activation='relu'),
        Dropout(rate=0.1),
        Dense(1080, activation='relu'),
        Dense(1)
    ])
    opt = keras.optimizers.Adam(lr=lr)
    # opt = keras.optimizers.SGD(lr=lr)
    model.compile(optimizer=opt, loss='mae', metrics=['mse', 'mae', 'mape'])
    return model


def data_preprocess(image):
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = vgg16.preprocess_input(input_arr)
    return input_arr


def load_train_set(paths):
    X = []
    y = []
    for folder_path in paths:
        print(folder_path)
        ground_truth_path = os.path.join(folder_path, 'ground_truth')
        for path in tqdm(glob.glob(os.path.join(ground_truth_path, '*.mat'))):
            mat = scipy.io.loadmat(path)
            y = np.append(y, mat.get('image_info')[0, 0][0, 0][1][0, 0])

        image_path = os.path.join(folder_path, 'images')
        for path in tqdm(glob.glob(os.path.join(image_path, '*.jpg'))):
            image = keras.preprocessing.image.load_img(path, target_size=(224, 224), interpolation='bicubic')
            X.append(data_preprocess(image))

    X = np.array(X)
    y = np.array(y)
    return X, y


def train_process(lr, patience, batch, epochs, both=True):
    # start time for log file
    now_time = time.strftime("%m%d%H%M%S", time.localtime())
    train_data_path = ['./part_A_final/train_data/']
    if both:
        train_data_path.append('./part_B_final/train_data/')

    X, y = load_train_set(train_data_path)

    aug = ImageDataGenerator()
    # aug = ImageDataGenerator(brightness_range=[0.2, 1.4], horizontal_flip=True, vertical_flip=True, rotation_range=20)

    model = VGGModel(lr=lr)
    model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint('./model/' + now_time + '_' + str(lr).replace('.', '-') + '_' + ('both_' if both else '') + str(batch) + '_model',
                         monitor='val_loss', mode='min', save_best_only=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    history = model.fit(aug.flow(X_train, y_train, batch_size=batch), validation_data=(X_test, y_test), epochs=epochs, callbacks=[es, mc])

    # kFold = KFold(n_splits=5)
    # i = 0
    # for trainIndex, testIndex in kFold.split(train_x, train_y):
    #     i += 1
    #     history = model.fit(aug.flow(train_x[trainIndex], train_y[trainIndex], batch_size=8), epochs=100,
    #                         validation_data=aug.flow(train_x[testIndex], train_y[testIndex]), callbacks=[es])
    #     print(history.history.keys())
    #
    #     # # summarize history for accuracy
    #     # plt.plot(history.history['accuracy'])
    #     # plt.title('model accuracy')
    #     # plt.ylabel('accuracy')
    #     # plt.xlabel('epoch')
    #     # plt.savefig('acc.png')
    #     # # summarize history for loss
    #
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['mse'], marker='.', label='mse')
    plt.plot(history.history['val_mse'], marker='x', color='green', label='val_mse')
    plt.legend()
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.title('model mse')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], marker='.', label='mae')
    plt.plot(history.history['val_mae'], marker='x', color='green', label='val_mae')
    plt.legend()
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.title('model mae')

    plt.subplot(1, 3, 3)
    plt.plot(history.history['mape'], marker='.', label='mape')
    plt.plot(history.history['val_mape'], marker='x', color='green', label='val_mape')
    plt.legend()
    plt.ylabel('mape')
    plt.xlabel('epoch')
    plt.title('model mape')
    plt.savefig('./log/' + now_time + '_' + str(lr).replace('.', '-') + '_' + ('both_' if both else '') + str(batch) + '_loss' + '.png')


if __name__ == "__main__":
    # lr_arr = [0.00001, 0.000001, 0.0000005, 0.0000001]
    lr_arr = [0.0000005]
    for lr in lr_arr:
        train_process(lr, patience=25, batch=8, epochs=1500)
        keras.backend.clear_session()
