import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("student_data.csv")  # ensure this has Attendance & Assignments columns

X = data[["AL501","AL502","AL503","AL504","AL505","AL506","AL507","AL508","Attendance","Assignments"]]
y = data["Performance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("student_model.pkl", "wb"))
print("Model trained and saved as student_model.pkl")
