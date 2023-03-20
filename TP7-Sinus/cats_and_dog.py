from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

X_train_gen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

X_test_gen = ImageDataGenerator(rescale = 1./255)

training_set = X_train_gen.flow_from_directory('TP7-Sinus/dataset/cats_and_dogs_filtered/train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

test_set = X_test_gen.flow_from_directory('TP7-Sinus/dataset/cats_and_dogs_filtered/validation',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

model = Sequential()

model.add(Conv2D(filters= 16, kernel_size=(3,3), activation ='relu', input_shape= (64,64,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(training_set,
epochs = 25,
validation_data = test_set
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()