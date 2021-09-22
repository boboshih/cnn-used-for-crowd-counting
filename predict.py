from model import data_preprocess
from tqdm import tqdm
import keras.preprocessing
import glob
import os
import numpy as np


def load_X(folder_path):
    X = []
    for path in tqdm(glob.glob(os.path.join(folder_path, '*.jpg'))):
        image = keras.preprocessing.image.load_img(path, target_size=(224, 224), interpolation='bicubic')
        X.append(data_preprocess(image))
    X = np.array(X)
    return X


if __name__ == "__main__":
    folder_path = './predict/'
    X = load_X(folder_path)

    model_path = './model/0107200137_5e-07_both_8_model'
    model = keras.models.load_model(model_path)

    result = model.predict(X)
    print(result)
