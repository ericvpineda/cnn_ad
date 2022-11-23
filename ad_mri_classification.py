from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import get_image_properties

batch_size = 32
features, target = get_image_properties('./dataset/train', (208, 208))
print('Completed import training data...')

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25)
train_size = len(features_train)
test_size = len(features_test)
# features_test, target_test = get_image_properties('./dataset/test', resize_dim)
# print('Completed import test data...')

features_train = features_train.reshape(len(features_train), 208, 208, 1)
features_test = features_test.reshape(len(features_test), 208, 208, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

target_train = to_categorical(target_train, 4)
target_test = to_categorical(target_test, 4)

# Create Convolution Neural Network 
model = Sequential()
# ----- Layer 1 -----
model.add(Conv2D(32, (3,3), input_shape=(208, 208, 1), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
# ----- Layer 2 -----
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# ----- Layer 3 -----
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
# ----- Layer 4 -----
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# ----- Layer 5 -----
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
# # ----- Layer 6 -----
# model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',
#     padding='same'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(BatchNormalization())

# Tranform 2D array --> 1D array 
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Create Deep Neural Network 
# model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
# model.add(BatchNormalization())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

opt = SGD(learning_rate=0.01, momentum=0.90)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(features_train, target_train, epochs=5, batch_size=batch_size, validation_data=(features_test, target_test), verbose=1)
results = model.evaluate(features_test, target_test, use_multiprocessing=True)
print(results)
