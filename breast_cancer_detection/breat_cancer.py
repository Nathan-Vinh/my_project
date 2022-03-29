import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


## DATASET CAN BE FOUND HERE :
## https://www.kaggle.com/yasserh/breast-cancer-dataset

df.info()
df.describe()

# heatmap
sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, cmap="coolwarm")

del df["id"]

df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == "M" else 0)


## create model

X = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, random_state=42)

# scale data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

# should give you 30
X_train_sc.shape[1]

# model and layers
model = Sequential()

model.add(Dense(32, input_dim=30, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# create early stop
er = EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=1)


model.fit(x=X_train_sc, y=y_train, epochs=1000, batch_size=256, callbacks=[er], validation_data=(X_test_sc, y_test))

# losses curves
df_losses = pd.DataFrame(model.history.history)
df_losses[["loss", "val_loss"]].plot()

# make predictions
y_pred = model.predict(X_test_sc)
y_pred = (y_pred > 0.5)

# classification report & confusion matrix
# should get 98%
print(classification_report(y_test, y_pred))

# should look like this:
# [87, 2]
# [1, 53]
confusion_matrix(y_test,y_pred)



