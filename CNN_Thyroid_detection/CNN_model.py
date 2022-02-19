# DATASET CAN BE FIND HERE:
# https://www.kaggle.com/tingzen/thyroid-for-pretraining
# once all images are in .png and in the right directory

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory( 'data\train', target_size=(200, 200),
    batch_size=128, class_mode='binary')

test_generator = train_datagen.flow_from_directory('data\test', target_size=(200, 200), 
    batch_size=128, class_mode='binary')


test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory('data\validation', target_size=(200, 200),
    batch_size=128, class_mode='binary')



# model creation

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(4,4), input_shape=(200,200,3), activation='relu' ))
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Conv2D(filters=64, kernel_size=(4,4), activation='relu' ))
#model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid'))

er = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_generator, epochs=10, callbacks=[er],
          validation_data=validation_generator)


# model evaluation

model.evaluate(train_generator) # 68%
model.evaluate(validation_generator) # 67%
model.evaluate(test_generator) # 65%


df_loss = pd.DataFrame(model.history.history)
df_loss[["loss", "val_loss"]].plot()

predictions = (model.predict(test_generator) > 0.5).astype('int32')
cm = confusion_matrix(test_generator[0][1], predictions)
sns.heatmap(cm, annot=True)

print(classification_report(test_generator[0][1], predictions)




