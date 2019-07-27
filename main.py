from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics import matthews_corrcoef


train_dir = 'train'
val_dir= 'val'
test_dir= 'test'
img_width, img_height = 350 ,350

input_shape = (img_width, img_height, 3)

epochs = 22
batch_size=25

nb_train_samples=1874

nb_validation_samples=560

nb_test_samples=456


model=Sequential()
model.add(Conv2D(64, (3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))

model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam',
             metrics=['accuracy'])


datagen = ImageDataGenerator(rescale=1. /255)


train_generator=datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


val_generator=datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


test_generator=datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


pred_generator=datagen.flow_from_directory(test_dir,
                                                     target_size=(350,350),
                                                     batch_size=13,
                                                     class_mode='categorical',shuffle=False)

model.summary()


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)


model.evaluate_generator(train_generator, nb_train_samples // batch_size)

model.evaluate_generator(val_generator, nb_validation_samples // batch_size)

model.evaluate_generator(test_generator, nb_test_samples // batch_size)

model.save_weights("project_model.h5")
