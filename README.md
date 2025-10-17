#BY DA25M001 NISHAN

# Imputation via Regression for Missing Data in Credit Risk Assessment

## Project Overview

This project tackles a common challenge in data science: handling missing data. Using the **UCI Credit Card Default Clients Dataset**, this analysis demonstrates and evaluates various techniques for imputing missing values. The primary goal is to determine how different imputation strategies—ranging from simple statistical methods to sophisticated regression models—impact the performance of a downstream classification task: predicting credit card payment defaults.

The project is structured as a complete data science workflow, including:
1.  **Data Preparation**: Simulating a real-world scenario by artificially introducing missing values.
2.  **Data Imputation**: Implementing four distinct strategies for handling missing data.
3.  **Model Training**: Building a logistic regression classifier on each of the prepared datasets.
4.  **Comparative Analysis**: Evaluating and comparing the performance of the models to provide a data-driven recommendation on the best imputation strategy.

## Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Imputation Strategies](#2-imputation-strategies)
  - [3. Model Training and Evaluation](#3-model-training-and-evaluation)
- [Key Findings](#key-findings)
- [Final Recommendation](#final-recommendation)
- [How to Run This Project](#how-to-run-this-project)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Execution](#execution)

## Dataset

The project uses the **UCI Credit Card Default Clients Dataset**. To simulate a more realistic data challenge, missing values (7%) were artificially introduced into the `AGE` and `BILL_AMT1` columns.

- **Source**: [Kaggle - Credit Card Default Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Target Variable**: `default.payment.next.month` (1 = default, 0 = no default)

## Methodology

The core of this project is a comparative study of four different methods for handling missing data.

### 1. Data Preprocessing
- Loaded the dataset and identified the target variable.
- Artificially introduced **Missing At Random (MAR)** values into the `AGE` and `BILL_AMT1` columns to create a controlled experimental setup.
- Visualized the missing data to confirm its distribution.

### 2. Imputation Strategies

Four datasets were created based on different handling strategies:

| Dataset | Strategy | Description |
| :--- | :--- | :--- |
| **Dataset A** | **Simple Imputation (Median)** | Missing values in each column were replaced with the column's **median**. This serves as a simple baseline. |
| **Dataset B** | **Linear Regression Imputation** | A **Linear Regression** model was trained to predict missing `AGE` values based on other features. This method assumes a linear relationship between variables. |
| **Dataset C** | **Non-Linear Regression Imputation** | A **K-Nearest Neighbors (KNN) Regressor** was used to predict missing `AGE` values. This method can capture complex, non-linear patterns in the data. |
| **Dataset D** | **Listwise Deletion** | All rows containing any missing values were completely removed from the dataset. This is a common but often problematic approach. |

### 3. Model Training and Evaluation

For each of the four datasets (A, B, C, and D):
1.  The data was split into an 80% training set and a 20% testing set.
2.  Features were standardized using `StandardScaler` to ensure the logistic regression model was not biased by feature scale.
3.  A **Logistic Regression** classifier was trained to predict credit card default.
4.  Model performance was evaluated on the test set using a full **Classification Report** (Accuracy, Precision, Recall, F1-Score) and a **Confusion Matrix**. The **weighted F1-Score** was used as the primary metric for comparison due to the imbalanced nature of the target variable.

## Key Findings

1.  **Imputation is Superior to Deletion**: All three imputation strategies significantly outperformed listwise deletion. The model trained on the deleted dataset (Model D) had the lowest F1-score, confirming that discarding rows leads to substantial information loss and a weaker, potentially biased model.

2.  **Regression-Based Methods Outperform Simple Imputation**: Both Linear and KNN Regression imputation (Models B and C) resulted in better-performing classifiers than simple median imputation (Model A). This highlights the value of using inter-variable relationships to make informed estimates for missing data, rather than using a simple statistic that distorts the data's variance.

3.  **Linear vs. Non-Linear Imputation**: The performance difference between Linear Regression and KNN Regression imputation was minimal for the final classification task. Although KNN did a better job of preserving the original data distribution visually, the model trained on the linearly imputed data performed slightly better. This may be due to the synergy between a linear imputation method and a linear classification model (Logistic Regression).

## Final Recommendation

**The recommended strategy is Linear Regression Imputation (Model B).**

-   **Performance**: It achieved the highest F1-score, indicating the best balance between precision and recall for this credit risk prediction task.
-   **Data Integrity**: This approach preserves the entire dataset, avoiding the bias and information loss associated with listwise deletion.
-   **Conceptual Soundness**: It is a robust method that leverages existing data patterns to make intelligent imputations, proving more effective than naive statistical fills.

This project demonstrates that a thoughtful approach to handling missing data is a critical step in the machine learning pipeline, with a direct and significant impact on final model performance.

## How to Run This Project

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab
- The required dataset file: `UCI_Credit_Card.csv`

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/DA5401_A6.git
    cd your-repo-name
    ```

2.  Install the required Python libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

### Execution

1.  Place the `UCI_Credit_Card.csv` file in the root directory of the project.
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```
3.  Open the `DA25M001_A6.ipynb` file and run the cells sequentially to reproduce the analysis.
