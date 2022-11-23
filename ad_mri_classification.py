from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam 
from keras.utils import to_categorical
from utils import get_image_properties

resize_dim = (208, 208)
features_train, target_train = get_image_properties('./dataset/train', resize_dim)
print('Completed import training data...')
features_test, target_test = get_image_properties('./dataset/test', resize_dim)
print('Completed import test data...')

features_train = features_train.reshape(len(features_train), 208, 208, 1)
features_test = features_test.reshape(len(features_test), 208, 208, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

target_train = to_categorical(target_train, 4)
target_test = to_categorical(target_test, 4)

model = Sequential()
# ----- Layer 1 -----
model.add(Conv2D(32, (3,3), input_shape=(208, 208, 1), activation='relu'))
model.add(BatchNormalization())
# ----- Layer 2 -----
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
# ----- Layer 3 -----
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
# ----- Layer 4 -----
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
# ----- Layer 5 -----
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
# ----- Layer 6 -----
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# Tranform 2D array --> 1D array 
model.add(Flatten())
# Create Deep Neural Network 
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(features_train, target_train, epochs=10, batch_size=64, validation_data=(features_test, target_test), verbose=1)
results = model.evaluate(features_test, target_test, use_multiprocessing=True)
print(results)







