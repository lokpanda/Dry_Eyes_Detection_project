import pandas as pd
import os
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


dataset_folder = r"C:\Users\Lenovo\Desktop\Dry_eyes_detection_project\datasets"
dataset_file = "preprocessed_dataset.csv"
dataset_path = os.path.join(dataset_folder, dataset_file)

# Read the dataset using the correct path
dataset = pd.read_csv(dataset_path)

X = dataset.iloc[:, :-2]
y = dataset.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Step 1: Hyperparameter Tuning for all models
param_grids = {
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly']
    }
}

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters for {name}:", best_params)

    # Model Training
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluation
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy for {name}:", test_accuracy)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = name

print("Best Model:", best_model)

joblib.dump(best_model, 'best_model.pkl')
