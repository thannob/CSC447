from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train = x_train[0:10000,:,:,:]
y_train = y_train[0:10000,:]
x_train = x_train/255
x_test = x_test/255

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu', padding='same',input_shape=(32,32,3)))
model.add(Conv2D(32,3,activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
             optimizer="Adam",
             metrics=["sparse_categorical_accuracy"])

model.fit(x_train, y_train, epochs=5)
model.save('final_model.keras')
