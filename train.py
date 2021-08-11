import tensorflow
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import plot_confusion_matrix
import itertools

path = "data/"
train_data_path = path + "train"
test_data_path = path + "test"
valid_data_path = path + "validation"

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)
train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(64, 64),
                                                    batch_size=10,
                                                    color_mode='grayscale',
                                                    class_mode='categorical', shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
valid_generator = validation_datagen.flow_from_directory(valid_data_path,
                                                         target_size=(64, 64),
                                                         batch_size=10,
                                                         color_mode='grayscale',
                                                         class_mode='categorical', shuffle=True)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_data_path,
                                                  target_size=(64, 64),
                                                  classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                                  batch_size=10,
                                                  color_mode='grayscale',
                                                  class_mode='categorical', shuffle=False)

classifier = Sequential()

classifier.add(Convolution2D(32, kernel_size=3, input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(128, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.6))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.0003)

classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

hist = classifier.fit(
    train_generator,
    steps_per_epoch=len(train_generator)//5,
    epochs=100,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)//5,
    # callbacks=[callback]
)

classifier.summary()

# Confusion Matrix
target_names = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
Y_pred = classifier.predict(test_generator, 1990)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = sklearn.metrics.confusion_matrix(test_generator.classes, y_pred)
print(cm)

# Classification Report
print('Classification Report')
labels = np.arange(10)
cr = sklearn.metrics.classification_report(test_generator.classes, y_pred, labels=labels, target_names=target_names)
print(cr)

# Training Accuracy & Loss
print(hist.history)
training_accuracy_arr = np.array(hist.history['accuracy'])
print("Training Accuracy = ", training_accuracy_arr)
loss_li = np.array(hist.history['loss'])
print("Training Loss = ", loss_li)
print("Average Training Accuracy = ", np.mean(hist.history['accuracy']))
print("Average Training Loss = ", np.mean(hist.history['loss']))

# Validation Accuracy & Loss
validation_accuracy_arr = np.array(hist.history['val_accuracy'])
print("Validation Accuracy = ", validation_accuracy_arr)
validation_loss_arr = np.array(hist.history['val_loss'])
print("Validation Loss = ", validation_loss_arr)
print("Average Validation Accuracy = ", np.mean(hist.history['val_accuracy']))
print("Average Validation Loss = ", np.mean(hist.history['val_loss']))

plt.figure(figsize=(12, 6))

# Plotting Training & Validation Loss
plt.subplot(121)
loss_train = hist.history['loss']
loss_val = hist.history['val_loss']
acc_train = hist.history['accuracy']
acc_val = hist.history['val_accuracy']
epochs = range(1, len(acc_train) + 1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Plotting Training & Validation Accuracy
plt.subplot(122)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.show()


# Plotting Confusion Matrix
def plotconfusionmatrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plotconfusionmatrix(cm, target_names)

# Plotting Classification Report

# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')
