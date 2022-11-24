from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.metrics import AUC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import get_image_properties
from tensorflow import argmax

batch_size = 32
resize_dim = (208, 208)

# Import training dataset
print("Testing training dataset...")
features, target = get_image_properties('./dataset/train', resize_dim)

# Split dataset into features and target labels
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25)
train_size = len(features_train)
test_size = len(features_test)

features_train = features_train.reshape(train_size, 208, 208, 1)
features_test = features_test.reshape(test_size, 208, 208, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

# Transform numerical labels to binary vectory array
target_train = to_categorical(target_train, 4)
target_test = to_categorical(target_test, 4)

# Create Convolution Neural Network 
model = Sequential()
# ----- Layer 1 -----
model.add(Conv2D(32, (3,3), input_shape=(208, 208, 1), activation='relu', kernel_initializer='he_uniform', padding='same'))
# ----- Layer 2 -----
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# ----- Layer 3 -----
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# ----- Layer 4 -----
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Tranform 2D array --> 1D array 
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Create Deep Neural Network 
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

# Note: Using AUC over standard accuracy metric
opt = SGD(learning_rate=0.01, momentum=0.95)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[AUC(name='auc')])
model.fit(features_train, target_train, epochs=5, batch_size=batch_size, validation_data=(features_test, target_test))

# Evaluate training dataset
results = model.evaluate(features_test, target_test, use_multiprocessing=True)
print(results)

# Evaluate test dataset 
print("Testing model with test dataset...\n")
test_features, test_target = get_image_properties('./dataset/test', resize_dim)
test_target_orig = test_target 

test_features = test_features.reshape(len(test_features), 208, 208, 1)
test_features = test_features.astype('float32')
test_target = to_categorical(test_target, 4)

# Get prediction labels of test dataset
predictions = model.predict(test_features[0], use_multiprocessing=True)

# Accuracy of test data
acc_score = accuracy_score(test_target, predictions.round())
print("Accuracy score: ", acc_score)

# Confusion matrix of test data
labels = argmax(predictions, axis=1)
conf_matrix = confusion_matrix(test_target_orig, labels)
print("Confusion matrix:\n", conf_matrix)






