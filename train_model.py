"""
Trains the same ANN architecture from the original notebook (ANN_model.ipynb)
and saves the artifacts app.py needs to serve predictions:
  - model.h5        (the trained Keras network)
  - scaler.pkl       (the fitted StandardScaler)
  - classes.json     (the label encoder's class order)

Run this once locally (python train_model.py) or let the Dockerfile run it
during the image build.
"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("Iris.csv")

X = df.drop(columns=["Species", "Id"])
y = df["Species"]

encoder = LabelEncoder()
y_int = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.2, random_state=42, stratify=y_int
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

model = Sequential([
    Dense(16, input_dim=4, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train_scaled, y_train_cat, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

loss, accuracy = model.evaluate(X_test_scaled, y_test_cat)
print(f"Test accuracy: {accuracy:.4f}")

model.save("model.keras")
joblib.dump(scaler, "scaler.pkl")
with open("classes.json", "w") as f:
    json.dump(list(encoder.classes_), f)

print("Saved model.keras, scaler.pkl, classes.json")
