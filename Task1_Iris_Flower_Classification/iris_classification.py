import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("Iris.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# Remove Id Column (if available)
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert Species into Numeric Values
le = LabelEncoder()
y = le.fit_transform(y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(
    df.iloc[:,0],
    df.iloc[:,1],
    c=le.fit_transform(df.iloc[:,-1])
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification Dataset")
plt.show()