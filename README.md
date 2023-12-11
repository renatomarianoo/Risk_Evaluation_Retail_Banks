# Risk Evaluation for Retail Banks

## Context
Welcome to the Capstone Project of of the Machine Learning Module! In this sprint, we embark on an exciting journey to develop a **risk evaluation service for retail banks**, leveraging the power of data science and machine learning.

You and your friend are launching a startup focused on providing risk evaluation as a service for retail banks. This proof-of-concept (POC) plan outlines the steps you will take to investigate, analyze, and build a solution using machine learning. The dataset for analysis is obtained from Home Credit Group.

## Necessary checks
1. The dataset contains relevant information about applicants' financial history and credit behavior.
2. The target variable for risk evaluation is clearly defined.

## Overall Objectives
1. Identify and understand key features in the dataset that may influence the risk evaluation for retail banks. 
2. Conduct exploratory data analysis (EDA), statistical inference, and machine learning model evaluation.
3. Develop and deploy multiple machine learning models to predict the target variable.
4. Provide a clear understanding of the predictive capabilities and limitations of the models.

## Plan Creation

### Step 1: Data Exploration and Cleaning
- **Objective:** Understand the structure of the dataset and handle missing values and outliers.
- **Tasks:**
  - Load the dataset and review the data description.
  - Check for missing values and decide on an imputation strategy.
  - Identify and handle outliers.
  - Perform basic summary statistics.

### Step 2: Exploratory Data Analysis (EDA)
- **Objective:** Uncover patterns, trends, and relationships within the dataset.
- **Tasks:**
  - Visualize the distribution of key variables.
  - Explore correlations between features and the target variable.
  - Identify potential features for prediction.
  - Evaluate the balance of the target variable.

### Step 3: Machine Learning Model Development
- **Objective:** Build and evaluate machine learning models for risk prediction.
- **Tasks:**
  - Split the dataset into training and testing sets.
  - Perform feature engineering and selection.
  - Implement multiple machine learning models.
  - Utilize hyperparameter tuning and model ensembling techniques.
  - Evaluate model performance using appropriate metrics.

### Step 4: Deployment to Google Cloud Platform
- **Objective:** Deploy machine learning models as an HTTP request.
- **Tasks:**
  - Choose a deployment option compatible with HTTP requests.
  - Deploy the models to Google Cloud Platform.
  - Test the deployed models with sample data.

### Step 5: Documentation and Suggestions for Improvement
- **Objective:** Provide clear explanations and enhancements for the analysis and models.
- **Tasks:**
  - Document the entire process in a notebook.
  - Identify areas for feature improvement or additional data collection.
  - Explore advanced machine learning techniques for future iterations.
  - Consider feedback from stakeholders and potential clients.

## Project Development
The project is composed by 5 notebooks including from EDA to Modeling. Some extra python files include utils, EDA, pipelines, feat-eng and, model-select. They were used to reduce the length of the notebooks/presentation time and to facilitate deployment of the model.


## Deployment 
Our model was successfully depolyed on Google Cloud and can be accessed through the link: https://risk-retail-banks-nhk7sfh42a-oe.a.run.app/docs

## Results
Our machine learning boosting models achieve results ROC AUC values around 0.76, having for the choosen/given threshold Recall slightly above 65%.

Some conclusions on the most important features:
- External sources play a key role in a way that if a client has bad sources they can be already denied a loan;
- Another vital feature is the ratio between the requested amount and the final credit amount (in the previous application);
- The Higher Education tends to push clients towards being good payers;
- Another feature that seems to define good payers is ratio between the Current debt and Current credit amount (in the Credit Bureau);