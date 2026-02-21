📂 Dataset Information

Dataset Name: Heart Disease Dataset

Target Column:

target

Target values:

0 → No Heart Disease

1 → Heart Disease

Features include:

age

sex

cp (chest pain)

trestbps (resting blood pressure)

chol (cholesterol)

thalach (maximum heart rate)

oldpeak

ca

thal

1️⃣ Train Decision Tree Classifier and Visualize Tree
Objective

Train a Decision Tree model and visualize decision structure.

Code
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

plt.figure(figsize=(15,8))
plot_tree(model, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.show()
Result

Decision Tree learned rules to classify heart disease.

2️⃣ Analyze Overfitting and Control Tree Depth
Objective

Prevent overfitting by limiting tree depth.

Code
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(train_acc, test_acc)
Explanation

Overfitting occurs when:

Training accuracy is very high

Testing accuracy is lower

Solution:

Use max_depth parameter
3️⃣ Train Random Forest and Compare Accuracy
Objective

Train Random Forest and compare with Decision Tree.

Code
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_accuracy = rf_model.score(X_test, y_test)

print("Random Forest Accuracy:", rf_accuracy)
Result

Random Forest gives higher accuracy than Decision Tree.

Reason:
Random Forest uses multiple trees.

4️⃣ Interpret Feature Importances
Objective

Identify most important features.

Code
import pandas as pd

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(feature_importance)
Result

Features like chest pain (cp), heart rate (thalach), and ca are important predictors.

5️⃣ Evaluate using Cross-Validation
Objective

Evaluate model using multiple data splits.

Code
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())
Result

Cross-validation gives reliable model accuracy.

📊 Comparison Summary
Model	Accuracy	Overfitting	Performance
Decision Tree	Moderate	Higher	Good
Random Forest	Higher	Lower	Better

Random Forest performs better.
