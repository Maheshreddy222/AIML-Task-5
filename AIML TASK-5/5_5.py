# Import libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load dataset
df = pd.read_csv("heart.csv")

# Step 2: Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# ============================
# Step 3: Decision Tree Cross-Validation
# ============================

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)

dt_scores = cross_val_score(
    dt_model,
    X,
    y,
    cv=5,              # 5-fold cross validation
    scoring="accuracy"
)

print("Decision Tree Cross-Validation Scores:", dt_scores)
print("Decision Tree Average Accuracy:", dt_scores.mean())


# ============================
# Step 4: Random Forest Cross-Validation
# ============================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nRandom Forest Cross-Validation Scores:", rf_scores)
print("Random Forest Average Accuracy:", rf_scores.mean())


# ============================
# Step 5: Compare models
# ============================

if rf_scores.mean() > dt_scores.mean():
    print("\nRandom Forest performs better")
else:
    print("\nDecision Tree performs better")