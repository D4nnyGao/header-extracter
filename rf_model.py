import pandas as pd
import joblib
import time
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler

# --- Define file paths and model name ---
INPUT_FILE = "final_labeled_dataset.csv"
MODEL_OUTPUT_NAME = 'random_forest_model.joblib'
FEATURES_JSON_PATH = 'model_features.json'
IMPORTANCE_PLOT_PATH = 'feature_importances.png'

# --- features for training ---
FEATURES_TO_USE = [
    'is_bold',
    'font_flag',
    'verb_count',
    'noun_count',
    'relative_length_ratio',
    'has_top_header_word',
    'has_header_pattern',
    'text_case'
]

# --- Define undersampling ratio ---
UNDERSAMPLING_RATIO = 15  # 15:1 ratio of non-headers to headers in training

# --- Define Hyperparameter Grid for GridSearchCV ---
# The script will search over all combinations of these parameters.
PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [12, 16, 24],
    'min_samples_leaf': [3, 5, 7],
    'criterion': ['gini', 'entropy']
}


def main():
    """Main function to train, evaluate, and export the model."""
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting model training script...")

    print(f"Loading data from '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f" ERROR: '{INPUT_FILE}' not found. Please run 'feature_engineering.py' first.")
        return

    # --- Feature Selection and Preparation ---
    final_features = []
    df_processed = df.copy()

    if 'text_case' in FEATURES_TO_USE:
        print("One-hot encoding 'text_case' feature...")
        df_processed = pd.get_dummies(df_processed, columns=['text_case'], drop_first=False, prefix='case')
        case_features = sorted([col for col in df_processed.columns if col.startswith('case_')])
        final_features.extend(case_features)
        FEATURES_TO_USE.remove('text_case')

    # Add remaining features and sort alphabetically for consistency
    final_features.extend(sorted(FEATURES_TO_USE))
    final_features.sort()

    X = df_processed[final_features]
    y = df_processed['is_header']

    print(f"\nTraining model with the following {len(final_features)} features:")
    for feature in final_features:
        print(f"- {feature}")

    # --- Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nData split successfully. Train: {len(X_train):,} | Test: {len(X_test):,}")

    # --- Apply Random Undersampling to the Training Set ---
    print(f"\nApplying Random Undersampling with a {UNDERSAMPLING_RATIO}:1 ratio...")
    minority_class_count = y_train.value_counts()[1]
    sampling_strategy = {0: minority_class_count * UNDERSAMPLING_RATIO, 1: minority_class_count}
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    print(f"Resampled training set shape: {X_train_res.shape}")

    # --- Hyperparameter Tuning with GridSearchCV ---
    print("\nStarting GridSearchCV to find the best hyperparameters...")
    # Using 'f1' for scoring is good for imbalanced datasets
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=PARAM_GRID,
        scoring='f1',
        cv=3, # 3-fold cross-validation
        verbose=1 # Shows progress
    )
    
    # Fit the grid search to the resampled training data
    grid_search.fit(X_train_res, y_train_res)

    print("\nGrid search complete.")
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # The best classifier is stored in 'best_estimator_'
    best_rf_clf = grid_search.best_estimator_

    # --- Evaluate the Model's Performance on the Test Set ---
    print("\n--- Model Evaluation (using the best model from GridSearch) ---")
    y_pred = best_rf_clf.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Header', 'Header']))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # --- Export Model and Feature Artifacts ---
    print(f"\n---  Exporting Artifacts ---")

    # Save the best trained model
    joblib.dump(best_rf_clf, MODEL_OUTPUT_NAME)
    print(f"Trained model saved as '{MODEL_OUTPUT_NAME}'")

    # Save the final feature list to a JSON file for inference consistency
    with open(FEATURES_JSON_PATH, 'w') as f:
        json.dump({'features': final_features}, f, indent=4)
    print(f"Feature list saved to '{FEATURES_JSON_PATH}'")

    # Create and save a plot of feature importances from the best model
    importances = pd.Series(best_rf_clf.feature_importances_, index=final_features)
    plt.figure(figsize=(10, 8))
    importances.sort_values().plot(kind='barh', color='skyblue')
    plt.title('Feature Importances for Random Forest Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH)
    print(f" Feature importance plot saved as '{IMPORTANCE_PLOT_PATH}'")
    
    end_time = time.time()
    print(f"\nScript finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()