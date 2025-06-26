# Task-03: Bank Marketing Decision Tree Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# 1. Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
df = pd.read_csv(url, sep=';')

# 2. Clean Data
df.drop(columns=["duration"], inplace=True)  # remove leakage feature

# 3. Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Split into features and target
X = df_encoded.drop(columns=['y_yes'])
y = df_encoded['y_yes']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train Decision Tree
clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
print("\n=== Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Plot top part of tree
plt.figure(figsize=(18, 8))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, max_depth=3, fontsize=10)
plt.title("Top Levels of Decision Tree")
plt.show()

# 9. Top 10 Important Features
importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\nTop 10 Important Features:\n", importances.sort_values(ascending=False).head(10))
