from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#evaluate model
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(clf, "iris_model.pkl")
print("Model trained and saved!")
