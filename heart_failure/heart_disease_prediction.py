import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  EarlyStopping

# DATASET CAN BE FIND HERE:
# https://www.kaggle.com/fedesoriano/heart-failure-prediction

df.info()
df.describe()
sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap="coolwarm")

df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "M" else 0)
df["ExerciseAngina"] = df["ExerciseAngina"].apply(lambda x: 1 if x == "Y" else 0)

def to_dummies(df, col):
    dummies = pd.get_dummies(df[col])
    df = df.drop(col,axis=1)
    df = pd.concat([df,dummies],axis=1)
    return df
  
df = to_dummies(df, "RestingECG")
df = to_dummies(df, "ChestPainType")
df = to_dummies(df, 'ST_Slope')


# model creation

X = df.drop("HeartDisease", axis=1).values
y = df["HeartDisease"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)


model = Sequential()

model.add(Dense(32, input_dim=18, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
  
  
er = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1)

model.fit(X_train_sc, y_train, epochs=500, batch_size=64,
          callbacks=[er], validation_data=(X_test_sc, y_test))

# model evaluation

model.evaluate(X_train_sc,y_train) #should get like 89%
model.evaluate(X_test_sc,y_test)

df_losses = pd.DataFrame(model.history.history)
df_losses[["loss", "val_loss"]].plot()

predictions = (model.predict(X_test_sc) > 0.5).astype('int32')

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, predictions))


  
  
  
  
  
  
  
