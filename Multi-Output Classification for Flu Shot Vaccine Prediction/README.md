# Multi-Output Classification for Flu Shot Vaccine Prediction
<img src="Image.jpg" alt="drawing" width="500"/>

## Project Overview
This project focuses on building a predictive model to determine the likelihood of individuals receiving flu vaccinations. Specifically, the goal is to predict two probabilities for each individual:
1. The probability of receiving the H1N1 vaccine.
2. The probability of receiving the seasonal flu vaccine.

The project leverages data from the National 2009 H1N1 Flu Survey, utilizing advanced machine learning techniques to address a multi-output classification problem. The primary performance metric used for evaluation is the Area Under the Receiver Operating Characteristic Curve (ROC AUC) for each target variable. The overall score is calculated as the average of the two ROC AUC scores.

## Dataset Description
The data was collected as part of the National 2009 H1N1 Flu Survey to monitor vaccination rates during the campaign. It comprises 26,707 individual records with approximately 35 features for each individual. These features provide insights into individuals':
- **Behavioral factors** (e.g., use of antiviral medication, wearing face masks, handwashing habits).
- **Demographic details** (e.g., age group, education level, health insurance status).
- **Opinions and perceptions** (e.g., views on vaccine effectiveness).

This rich dataset provides a foundation for building predictive models to address the multi-output classification problem.

## Methodology
### Multi-Output Classification Approaches
Several approaches exist for handling multi-output classification problems. Below are some examples:

#### 1. MultiOutputClassifier
- **What It Does**: Wraps a single-output classifier to handle multiple target variables by creating an independent classifier for each target.
- **How It Works**:
  - For `n` target variables, `n` independent classifiers are trained.
  - Each classifier focuses solely on one output variable.

#### 2. Chain Classifiers (ClassifierChain)
- **What It Does**: Models the dependencies between output variables by training classifiers sequentially, where each classifier uses the predictions of previous classifiers as additional features.
- **How It Works**:
  - For `n` target variables, `n` classifiers are trained in sequence.
  - Each classifier is conditioned on the outputs of earlier classifiers in the chain.

Given that the target outputs in this dataset are not closely related, the **MultiOutputClassifier** approach was selected for this project.

## Machine learning algorithms used
The following machine learning classifiers were used to build the predictive models:
1. **Decision Trees:**
      Decision Trees are simple to understand and interpret. They work well with small to medium-sized datasets and handle categorical features effectively. 
They provide feature importance scores, which help in identifying key variables contributing to the prediction task. This insight was valuable in understanding the dataset.

3. **Logistic Regression:**
   This methods is simple and efficient for binary classification tasks, with interpretability and low computational requirements. It assumes a linear relationship between features and target variables.
Logistic regressions erved as a strong baseline model, providing insights into feature relationships and setting a benchmark for performance.

5. **Random Forest Classifier:**
   It is an ensemble method that combines multiple decision trees to improve robustness and accuracy. It handles overfitting better than individual decision trees and works well with both categorical and numerical features.
The Random Forest Classifier's robust feature importance estimation helped refine the feature set and improve overall model stability and performance.

6. **XGBoost Classifier (XGB):**
   It is a gradient boosting framework that is highly efficient, flexible, and accurate. It performs exceptionally well on structured/tabular datasets. XGBoost incorporates regularization techniques to prevent overfitting and supports advanced tuning options.
  XGBoost provided the similar AUROC scores as that of the Random Forest classifiers. Its ability to model complex interactions between features made it an excellent choice for this analysis.

## Project Workflow
**Step 1 : Exploratory Data Analysis (EDA)**
- To better understand the dataset, an extensive Exploratory Data Analysis (EDA) was performed:
- Several methods are used to analyze feature label relationship. For this project Correlation analysis methods is used to analyze this relationship  
  - **Pearson Correlation:** Measures the linear relationship between two numerical features, providing a coefficient between -1 (strong negative correlation) and 1 (strong positive correlation).
  - **Spearman's Rank Correlation:** Suitable for non-normally distributed data, analyzes the monotonic relationship between variables based on their ranks.

**Step 2 : Data Preparation**
- **Correlation Analysis**: Variables with little or no correlation with both target variables were removed.
- **Data Cleaning**: Missing values were handled using appropriate imputation techniques.
- **Data Splitting**: The dataset was split into training, validation, and test sets for robust model evaluation.

**Step 3 : Model Training and Validation**
- Following classification methods were implemented and evaluated:
  1. Decision Trees
  2. Logistic Regression
  3. Random Forest Classifier
  4. XGBoost Classifier (XGB)

**Step 4 : Hyperparameter Tuning**
- After analyzing the baseline models, hyperparameter optimization was performed on Random Forest classfier to further improve the AUROC scores.
- Techniques such as grid search and random search were employed to fine-tune model parameters.

**Step 5 : Results**
- The project was successfully completed as part of the Driven Data competition. The final model achieved a rank in the **top 10%** on the competition leaderboard, showcasing its effectiveness in predicting the likelihood of individuals receiving H1N1 and seasonal flu vaccines.

## Key Takeaways
- MultiOutputClassifier proved to be an effective approach for this multi-output classification task, given the lack of strong dependency between target variables.
- EDA and feature engineering played a critical role in improving model performance by identifying key features and eliminating irrelevant ones.
- Random Forest Classfier provided superior performance compared to other classifiers and used this model for final submission.

## Future Work
- Incorporate additional external datasets to enhance the model's generalizability.
- Investigate feature interactions to further improve the predictive performance of the models.

---

Thank you for reviewing this project. For further inquiries or collaboration opportunities, feel free to reach out!

