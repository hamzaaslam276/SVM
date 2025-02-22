# Handwritten Digit Classification using SVM

## Overview
This project demonstrates how to classify handwritten digits from the Fashion-MNIST dataset using a **Support Vector Machine (SVM)** model. The dataset consists of **28x28 grayscale images** of various fashion items, represented as numerical pixel values.

## Dataset
The dataset used is **fashion-mnist_test.csv**, which contains 10,000 rows and 785 columns:
- The first column (`label`) represents the class of the fashion item (0-9).
- The remaining 784 columns represent pixel intensity values (0-255).

## Libraries Used
- **NumPy**: For numerical operations
- **Pandas**: For data manipulation
- **Matplotlib & Seaborn**: For visualization
- **Scikit-learn**: For preprocessing, model training, and evaluation

## Implementation Steps
1. **Load the Dataset**: Read the dataset using Pandas.
2. **Data Preprocessing**:
   - Extract features (`X`) and labels (`y`).
   - Split the data into training (80%) and testing (20%) sets.
   - Standardize feature values using `StandardScaler`.
3. **Model Training**:
   - Train an **SVM model with a linear kernel** using Scikit-learn’s `SVC`.
4. **Model Evaluation**:
   - Predict labels for the test set.
   - Compute model accuracy.
   - Generate a classification report (precision, recall, f1-score).
   - Visualize the confusion matrix using Matplotlib.

## Results
- **Model Accuracy**: **79.30%**
- **Classification Report**:
  ```
              precision    recall  f1-score   support

           0       0.67      0.73      0.70       192
           1       0.96      0.96      0.96       192
           2       0.63      0.69      0.66       212
           3       0.85      0.85      0.85       205
           4       0.72      0.73      0.72       204
           5       0.84      0.92      0.88       193
           6       0.50      0.42      0.46       200
           7       0.88      0.85      0.86       206
           8       0.95      0.90      0.92       196
           9       0.94      0.89      0.92       200

    accuracy                           0.79      2000
   macro avg       0.79      0.79      0.79      2000
  weighted avg       0.79      0.79      0.79      2000
  ```

## Visualizations
- Confusion Matrix:
  ![Confusion Matrix](confusion_matrix.png)

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fashion-mnist-svm.git
   ```
2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the script:
   ```bash
   python fashion_mnist_svm.py
   ```

## Future Improvements
- Use **Convolutional Neural Networks (CNNs)** for improved accuracy.
- Experiment with different SVM kernels (e.g., RBF, polynomial).
- Perform hyperparameter tuning for optimal performance.
