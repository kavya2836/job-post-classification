import pandas as pd
import re
import nltk
import logging
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import joblib

# Setup logging
logging.basicConfig(
    filename='training.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download and set up NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load dataset
def load_dataset():
    try:
        df = pd.read_csv("datasets/post_updated12.csv", encoding='ISO-8859-1')
        df.columns = df.columns.str.lower()
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

df = load_dataset()

# Text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Clean dataset
df['cleaned_description'] = df['job_description'].apply(clean_text)
df['combined_features'] = (
    df['company_name'] + " " + df['job_title'] + " " +
    df['cleaned_description'] + " " + df['job_location'] + " " + df['job_experience']
)

# Ensure authenticity column exists
if 'authenticity' not in df.columns:
    logging.error("Column 'authenticity' is missing from the dataset!")
    raise ValueError("Column 'authenticity' is missing from the dataset!")

# Remove missing values
df.dropna(subset=['combined_features', 'authenticity'], inplace=True)
X = df['combined_features']
y = df['authenticity']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize features
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Apply SMOTE only on the training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

# Train RandomForest model
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train_res, y_train_res)

best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
logging.info(f"RandomForest Model Accuracy: {round(accuracy_rf * 100, 2)}%")
print(f"RandomForest Model Accuracy: {round(accuracy_rf * 100, 2)}%")
print("\nRandomForest Classification Report:\n", classification_report(y_test, y_pred_rf, digits=2))

# Train SVM model
svm_model = SVC(probability=True, random_state=42)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, verbose=1, n_jobs=-1)
grid_search_svm.fit(X_train_res, y_train_res)

best_svm_model = grid_search_svm.best_estimator_
y_pred_svm = best_svm_model.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
logging.info(f"SVM Model Accuracy: {round(accuracy_svm * 100, 2)}%")
print(f"SVM Model Accuracy: {round(accuracy_svm * 100, 2)}%")
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm, digits=2))

# Save accuracies to a JSON file
accuracy_data = {
    "random_forest_accuracy": round(accuracy_rf * 100, 2),
    "svm_accuracy": round(accuracy_svm * 100, 2)
}
with open("models/accuracy.json", "w") as f:
    json.dump(accuracy_data, f)
logging.info("Model accuracies saved successfully.")



import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

def plot_learning_curve(estimator, title, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curve for Random Forest
plot_learning_curve(best_rf_model, "Learning Curve - Random Forest", X_train_res, y_train_res)
plt.savefig("models/learning_curve_rf.png")  # Save image
plt.show()

# Plot learning curve for SVM
plot_learning_curve(best_svm_model, "Learning Curve - SVM", X_train_res, y_train_res)
plt.savefig("models/learning_curve_svm.png")  # Save image
plt.show()
