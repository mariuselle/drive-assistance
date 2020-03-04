from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from glob import glob
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
import cv2
from keras.callbacks import ModelCheckpoint

from helpers import flip_image 


VEHICLE_DIR = 'data/train/vehicles/'
NON_VEHICLE_DIR = 'data/train/non-vehicles/'

class VehicleModel:

    def __init__(self, model_name):
        self.__model_name = model_name

    def create_FCNN(self, input_shape=(720, 1280, 3)):
        model = Sequential()

        # Center and normalize our data
        model.add(Lambda(lambda x: x / 255., input_shape=input_shape, output_shape=input_shape))
        
        # Block nr.0
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='cv0', input_shape=input_shape, padding='same'))
        model.add(Dropout(0.5))

        # Block nr.1
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='cv1', padding='same'))
        model.add(Dropout(0.5))

        # Block nr.2
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='cv2', padding='same'))
        model.add(MaxPooling2D(pool_size=(8, 8)))
        model.add(Dropout(0.5))

        # Binary 'classifier'
        model.add(Conv2D(filters=1, kernel_size=(8, 8), name='fcn', activation='sigmoid'))

        return model, self.__model_name


    def _generator(self, images, batch_size=32, use_flips=False, resize=False):
        images_count = len(images)
        
        while True:
            shuffle(images)
            for index in range(0, images_count, batch_size):
                batch_images = images[index : index + batch_size]
                x_train = []
                y_train = []

                for batch_image in batch_images:
                    y = float(batch_image[1])
                    filename = batch_image[0]
                    
                    image = cv2.imread(filename)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    x_train.append(image)
                    y_train.append(y)

                    if use_flips:
                        fliped_image = flip_image(image)
                        x_train.append(fliped_image)
                        y_train.append(y)
                    
                x_train = np.array(x_train)
                y_train = np.expand_dims(y_train, axis=1)

                yield shuffle(x_train, y_train)

    def _create_dataset(self, x, y):
        
        assert len(x) == len(y)

        return [ (x[i], y[i]) for i in range(len(x)) ]

    def _get_data(self, filename, create=False):

        filename = 'data/structures/{}.h5'.format(filename)

        if create:
            vehicles = glob('{}*/*.png'.format(VEHICLE_DIR), recursive=True)
            non_vehicles = glob('{}*/*.png'.format(NON_VEHICLE_DIR), recursive=True)
            
            images = vehicles + non_vehicles
            y = np.concatenate((np.ones(len(vehicles)), np.zeros(len(non_vehicles))))
            images, y = shuffle(images, y)

            x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

            data = { 'xTrain': x_train,  'xValidation': x_val,   'xTest': x_test,
                    'yTrain': y_train,  'yValidation': y_val,   'yTest': y_test }

            pickle.dump(data, open(filename, 'wb'))

            return x_train, x_val, x_test, y_train, y_val, y_test
        
        with open(filename, mode='rb') as f:
            data = pickle.load(f)

            xTrain = data['xTrain']
            xValidation = data['xValidation']
            xTest = data['xTest']
            yTrain = data['yTrain']
            yValidation = data['yValidation']
            yTest = data['yTest']

            return xTrain, xValidation, xTest, yTrain, yValidation, yTest

    def train(self, create=False, batch_size=32, use_flips=True, epoch_count=3, proceed=True):
        x_train, x_val, x_test, y_train, y_val, y_test = self._get_data(self.__model_name, create)

        train_data = self._create_dataset(x_train, y_train)
        validation_data = self._create_dataset(x_val, y_val)

        if use_flips:
            inflate_factor = 2
        else:
            inflate_factor = 1

        steps_per_epoch = len(train_data) * inflate_factor / batch_size
        print('Steps per epoch: {}'.format(steps_per_epoch))

        validation_steps = len(validation_data) * inflate_factor / batch_size
        print('Validation steps per epoch: {}'.format(validation_steps))

        if proceed:
            source_model, model_name = self.create_FCNN(input_shape=(64, 64, 3))

            x = source_model.output
            x = Flatten()(x)
            model = Model(inputs = source_model.input, outputs = x)

            print(model.summary())

            model.compile(optimizer = 'adam', loss='mse', metrics=['accuracy'])

            train_gen = self._generator(images = train_data, use_flips=use_flips)
            validation_gen = self._generator(images = validation_data, use_flips=use_flips)


            filepath = model_name + '.h5'

            checkpoint = ModelCheckpoint(filepath = filepath, monitor='val_acc', verbose = 0, save_best_only=True)

            model.fit_generator(train_gen, steps_per_epoch = steps_per_epoch, 
                                validation_data = validation_gen, validation_steps=validation_steps,
                                epochs=epoch_count, callbacks=[checkpoint])

            
            print('Training complete..')
            
            print('Evaluating accuracy..')
            test_data = self._create_dataset(x_test, y_test)
            test_gen = self._generator(images = test_data, use_flips=False)
            test_steps = len(test_data) / batch_size
            accuracy = model.evaluate_generator(generator = test_gen, steps = test_steps)
            print('Accuracy:', accuracy)
        

            

    