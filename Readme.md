# Diabetes Prediction: Comparison of KNN and Logistic Regression Models

This project aims to predict diabetes using two machine learning models: K-Nearest Neighbors (KNN) and Logistic Regression. The dataset used for this project is the Pima Indians Diabetes Database, which contains medical diagnostic measurements.

## Notebooks Overview

### 1. Data Preprocessing
- **File**: `data_preprocesisng.ipynb`
- **Description**: 
  - Loads the dataset (`diabetes.csv`).
  - Cleans the data by removing invalid rows and replacing zero values in key columns with median values.
  - Splits the dataset into training and validation sets.

### 2. KNN Model
- **File**: `models/knn.ipynb`
- **Description**: 
  - Trains a KNN model using the training dataset.
  - Uses the elbow method to determine the optimal number of neighbors (`k`).
  - Saves the trained model as `knn_model.pkl`.

### 3. Logistic Regression Model
- **File**: `models/logistic_regression.ipynb`
- **Description**: 
  - Trains a Logistic Regression model using the training dataset.
  - Evaluates the model's performance and saves it as `logistic_regression_model.pkl`.

### 4. Model Comparison
- **File**: `model_comparison.ipynb`
- **Description**: 
  - Loads the trained models and validation dataset.
  - Compares the performance of KNN and Logistic Regression models on the validation set.
  - Reports accuracy and provides insights into the results.

## Results

| Model  | Testing Accuracy | Validation Accuracy |
|--------|------------------|---------------------|
| LogReg | 76%              | 76%                 |
| KNN    | 73%              | 69%                 |

Logistic Regression slightly outperforms KNN in both testing and validation accuracy, making it the preferred model for this dataset.

## How to Run

1. **Set up the environment**:
   - Use the provided virtual environment in the `env/` folder or create a new one.
   - Install required dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the notebooks**:
   - Start with `data_preprocesisng.ipynb` to preprocess the data.
   - Train the models using `models/knn.ipynb` and `models/logistic_regression.ipynb`.
   - Compare the models using `model_comparison.ipynb`.

3. **View results**:
   - Check the accuracy and insights provided in the `model_comparison.ipynb` notebook.

## Dataset

The dataset is located in the `data/diabetes.csv` file. It contains the following columns:
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (1 = Diabetes, 0 = No Diabetes)


