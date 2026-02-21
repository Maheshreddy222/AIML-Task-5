# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv("heart.csv")

# Step 2: Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# Step 4: Train Decision Tree
# ============================

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Predict and calculate accuracy
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_accuracy)


# ============================
# Step 5: Train Random Forest
# ============================

rf_model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=3,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predict and calculate accuracy
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)


# ============================
# Step 6: Compare Results
# ============================

print("\nComparison:")
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

if rf_accuracy > dt_accuracy:
    print("Random Forest performs better")
else:
    print("Decision Tree performs better")