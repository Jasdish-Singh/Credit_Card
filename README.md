Credit Card Fraud Detection Report
Project Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset contains credit card transactions from European cardholders in September 2013. Out of 284,807 transactions, only 492 are fraudulent, making the dataset highly imbalanced.

Dataset Description
The dataset includes the following features:

Time: The elapsed seconds since the first transaction.

Amount: The transaction value.

The target variable, Class, indicates whether a transaction is fraudulent (1) or legitimate (0).

Features are derived through PCA transformation for dimensionality reduction.

Data Preprocessing
Checked for missing values, and none were found.

Visualized the class distribution to highlight the imbalance between legitimate and fraudulent transactions.

Standardized the Time and Amount features using StandardScaler.

Split the dataset into training (80%) and testing (20%) sets.

Models Used
Logistic Regression: A baseline linear model for binary classification.

Random Forest Classifier: An ensemble model that works well for imbalanced datasets.

Evaluation Metrics
Due to the class imbalance, accuracy is not a reliable metric. Instead, the following evaluation metrics are used:

Confusion Matrix: To evaluate the true positives, false positives, true negatives, and false negatives.

Precision-Recall Curve: To assess the precision and recall at various thresholds.

ROC-AUC Score: To measure the model's ability to distinguish between the classes.

Results
Logistic Regression Performance:
Confusion Matrix: Displays the model's performance in terms of true positives, false positives, true negatives, and false negatives.

Classification Report: Provides precision, recall, and F1 score for both classes.

ROC AUC Score: Measures the overall ability of the model to discriminate between fraudulent and legitimate transactions.

AUPRC: Area Under Precision-Recall Curve to handle the class imbalance better.

Random Forest Performance:
Confusion Matrix: Evaluates the model's performance with respect to both legitimate and fraudulent transactions.

Classification Report: Includes precision, recall, and F1 score for each class.

Precision-Recall Curves: Visualizes the trade-off between precision and recall at various thresholds.

Visualizations
Class Distribution Plot: Displays the imbalance between fraudulent and legitimate transactions.

Feature Correlation Heatmap: Shows correlations between features to understand their relationships.

Confusion Matrices: Visual comparisons of both models' performance.

Precision-Recall Curves: Plots showing the precision-recall trade-offs for both models.

How to Run the Project
Install the necessary dependencies:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn
Place the creditcard.csv dataset in the working directory.

Run the script:

bash
Copy
Edit
python creditcard_fraud.py
Future Enhancements
Explore additional classification models such as XGBoost and SVM to improve performance.

Implement oversampling or undersampling techniques (e.g., SMOTE) to handle class imbalance more effectively.

Explore real-time fraud detection by applying the model to streaming data, making it suitable for production environments.

