#Necessary Imports
import argparse, os
import boto3
import numpy as np
import pandas as pd
import sagemaker
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import backend as K

import configparser

config = configparser.ConfigParser()
config.read('env.ini')

# This is an object that represents the SageMaker session that we are currently operating in. This
# object contains some useful information that we will need to access later such as our region.
# session = sagemaker.Session()

# SageMaker default SM_MODEL_DIR=/opt/ml/model
if os.getenv("SM_MODEL_DIR") is None:
    os.environ["SM_MODEL_DIR"] = config['SM']['SM_MODEL_DIR'] + '/model'

# SageMaker default SM_OUTPUT_DATA_DIR=/opt/ml/output
if os.getenv("SM_OUTPUT_DATA_DIR") is None:
    os.environ["SM_OUTPUT_DATA_DIR"] = config['SM']['SM_OUTPUT_DATA_DIR'] + '/output'

# SageMaker default SM_CHANNEL_TRAINING=/opt/ml/input/data/training
if os.getenv("SM_CHANNEL_TRAINING") is None:
    os.environ["SM_CHANNEL_TRAINING"] = config['SM']['SM_CHANNEL_TRAINING'] + '/data'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    lr         = args.learning_rate
    model_dir  = args.model_dir
    training_dir   = args.train
    
    ############
    #Reading in data
    ############
    df = pd.read_csv('data/train.csv',sep=',')
    
    ############
    #Preprocessing data
    ############
    X = df.drop(['Species'],axis=1)
    y = df['Species']
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler() #scaling X data before model training
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    
    ###########
    #Model Building
    ###########
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_shape=[4,]))
    model.add(Dropout(.3))
    model.add(Dense(units=3,activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',metrics=['accuracy'])
    early_stop = EarlyStopping(patience=10)
    model.fit(x=scaled_X_train, 
          y=y_train, 
          epochs=epochs,
          validation_data=(scaled_X_test, y_test), verbose=1 ,callbacks=[early_stop])
    
    #Storing model artifacts
    model.save(os.path.join(model_dir, '000000001'), 'my_model.h5')