import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Step 1: Load the dataset
file_path = "./a9a-1.txt"  # Replace with the correct path to the dataset
X_sparse, y = load_svmlight_file(file_path)
X = np.asarray(X_sparse.todense())  # Convert to a NumPy array

# Step 2: Split the data into train and test parts (30% test data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

### Gaussian Naive Bayes Classifier
print("\n--- Gaussian Naive Bayes Classifier ---")
pipe_gnb = Pipeline(
    [
        ("scaler", StandardScaler()),  # Scaling features for Gaussian Naive Bayes
        ("gnb", GaussianNB()),
    ]
)

# Fit the pipeline to the training data
pipe_gnb.fit(X_train, y_train)

# Evaluate the pipeline on the test set
accuracy_gnb = pipe_gnb.score(X_test, y_test)
print(f"Classification accuracy of Gaussian Naive Bayes: {accuracy_gnb:.4f}")

### Decision Tree Classifier
print("\n--- Decision Tree Classifier ---")
pipe_dt = Pipeline([("scaler", StandardScaler()), ("dt", DecisionTreeClassifier())])

# Define the parameter grid for Decision Tree
param_grid_dt = {"dt__criterion": ["gini", "entropy"], "dt__max_depth": [10, 50, 100]}

# GridSearchCV for Decision Tree
grid_search_dt = GridSearchCV(
    pipe_dt,
    param_grid_dt,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)

# Fit GridSearchCV on the training data
grid_search_dt.fit(X_train, y_train)

# Extract the best model and parameters
best_model_dt = grid_search_dt.best_estimator_
best_params_dt = grid_search_dt.best_params_
print(f"Best Decision Tree parameters: {best_params_dt}")

# Evaluate the best Decision Tree model on the test set
accuracy_dt = best_model_dt.score(X_test, y_test)
print(f"Classification accuracy of Decision Tree: {accuracy_dt:.4f}")

### Support Vector Machine (SVM) Classifier
print("\n--- SVM Classifier ---")
pipe_svm = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])

# Define the parameter grid for SVM
param_grid_svm = {
    "svm__kernel": ["linear", "poly", "rbf"],
    "svm__degree": [2, 3],  # Polynomial kernel degrees
    "svm__gamma": [0.001, 0.1, 2],  # Gamma for rbf/poly
    "svm__C": [1, 10],  # Regularization parameter
}

# GridSearchCV for SVM
grid_search_svm = GridSearchCV(
    pipe_svm,
    param_grid_svm,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)

# Fit GridSearchCV on the training data
grid_search_svm.fit(X_train, y_train)

# Extract the best model and parameters
best_model_svm = grid_search_svm.best_estimator_
best_params_svm = grid_search_svm.best_params_
print(f"Best SVM parameters: {best_params_svm}")

# Evaluate the best SVM model on the test set
accuracy_svm = best_model_svm.score(X_test, y_test)
print(f"Classification accuracy of SVM: {accuracy_svm:.4f}")
