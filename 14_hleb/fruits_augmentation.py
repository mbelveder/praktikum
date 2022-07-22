from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def load_train(path):
    
    datagen_train = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True
        # height_shift_range=0.2
        )
    
    train_datagen_flow = datagen_train.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)

    return train_datagen_flow

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=6, input_shape=input_shape, 
              kernel_size=(5, 5), padding='same', activation='tanh'))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
              padding='valid', activation='tanh'))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
              padding='valid', activation='tanh'))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None))
    model.add(Flatten())
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(12, activation='softmax'))

    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=20,
               steps_per_epoch=None, validation_steps=None):

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model