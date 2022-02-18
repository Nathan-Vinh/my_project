# INSTEAD OF PREDICTING HEART DISEASE
# WE TRY TO PREDICT THE TYPE OF CHEST PAIN


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  EarlyStopping
from tensorflow.keras.utils import to_categorical

# preprocess
df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "M" else 0)
df["ExerciseAngina"] = df["ExerciseAngina"].apply(lambda x: 1 if x == "Y" else 0)

def to_dummies(df, col):
    dummies = pd.get_dummies(df[col])
    df = df.drop(col ,axis=1)
    df = pd.concat([df,dummies],axis=1)
    return df
   
df = to_dummies(df, 'RestingECG')
df = to_dummies(df, 'ST_Slope')

chestpain_dict = {"ASY" : 0, "NAP" : 1,"ATA" : 2, "TA" : 3}
def pain_type(x):
    return chestpain_dict[x]

df["ChestPainType"] = df["ChestPainType"].apply(pain_type)


# model creation

X = df.drop("ChestPainType", axis=1).values
y = df["ChestPainType"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=42, stratify=y)

scaler = MinMaxScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)


model = Sequential()

model.add(Dense(16, input_dim=15, activation="relu"))
model.add(Dropout(0.2))
#model.add(Dense(8, activation="relu"))
#model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", 
              metrics=["accuracy"])


er = EarlyStopping(monitor="val_loss", mode="min", patience=25, verbose=1)

model.fit(X_train_sc, y_train, epochs=500, batch_size=256, callbacks=[er],
          validation_data=(X_test_sc, y_test))


# model evaluation

df_losses = pd.DataFrame(model.history.history)
df_losses[["loss", "val_loss"]].plot()

model.evaluate(X_train_sc, y_train) #should get something like 63%
model.evaluate(X_test_sc, y_test)

predictions = model.predict(X_test_sc)


cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
sns.heatmap(cm, annot=True)


print(classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))
