import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Data Preprocessing
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    X = data.drop('clicked', axis=1)
    y = data['clicked']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Feature Engineering
def add_feature_interactions(data):
    data['interaction'] = data['feature1'] * data['feature2']
    return data

def engineer_features(data):
    data = add_feature_interactions(data)
    return data

# Model Training
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

# Clustering
def apply_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters

# Evaluation
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    
    return accuracy, report, matrix

def print_evaluation(accuracy, report, matrix):
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report: \n{report}")
    print(f"Confusion Matrix: \n{matrix}")

# Hyperparameter Tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_

# Main function
def main():
    data_file = 'ads_click_data.csv'
    
    # Load and preprocess the data
    data = load_data(data_file)
    data = engineer_features(data)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    # Predict and evaluate
    y_pred_rf = predict(rf_model, X_test)
    accuracy_rf, report_rf, matrix_rf = evaluate_model(y_test, y_pred_rf)
    print("Random Forest Evaluation:")
    print_evaluation(accuracy_rf, report_rf, matrix_rf)
    
    # Apply K-means clustering
    print("Applying K-means Clustering...")
    clusters = apply_kmeans(X_train)
    
    # Hyperparameter Tuning
    print("Tuning Hyperparameters...")
    best_params = tune_hyperparameters(X_train, y_train)
    print(f"Best Parameters: {best_params}")

if __name__ == '__main__':
    main()
