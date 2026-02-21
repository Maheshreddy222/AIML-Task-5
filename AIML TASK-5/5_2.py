# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
# Step 4: Train Decision Tree WITHOUT depth limit (may overfit)
# ============================

model_overfit = DecisionTreeClassifier(random_state=42)

model_overfit.fit(X_train, y_train)

# Check accuracy
train_acc_overfit = accuracy_score(y_train, model_overfit.predict(X_train))
test_acc_overfit = accuracy_score(y_test, model_overfit.predict(X_test))

print("Overfitting Model:")
print("Training Accuracy:", train_acc_overfit)
print("Testing Accuracy:", test_acc_overfit)


# ============================
# Step 5: Train Decision Tree WITH depth limit (control overfitting)
# ============================

model_control = DecisionTreeClassifier(max_depth=3, random_state=42)

model_control.fit(X_train, y_train)

# Check accuracy
train_acc_control = accuracy_score(y_train, model_control.predict(X_train))
test_acc_control = accuracy_score(y_test, model_control.predict(X_test))

print("\nControlled Model (max_depth=3):")
print("Training Accuracy:", train_acc_control)
print("Testing Accuracy:", test_acc_control)


# ============================
# Step 6: Visualize controlled tree
# ============================

plt.figure(figsize=(15,8))

plot_tree(
    model_control,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True
)

plt.title("Decision Tree with Controlled Depth")
plt.show()


# ============================
# Step 7: Compare accuracy vs depth
# ============================

depths = range(1, 10)

train_scores = []
test_scores = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)

    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.plot(depths, train_scores, label="Training Accuracy")
plt.plot(depths, test_scores, label="Testing Accuracy")

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis")
plt.legend()

plt.show()