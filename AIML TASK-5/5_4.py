# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load dataset
df = pd.read_csv("heart.csv")

# Step 2: Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Get feature importances
importances = model.feature_importances_

# Create DataFrame for visualization
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)

# Print feature importance
print(feature_importance_df)

# Step 6: Plot feature importance
plt.figure(figsize=(10,6))

plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"])

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.show()