# Multiclass Classification for Predicting Obesity Risk  

## Project Overview  
This project was part of the Kaggle competition, where the objective was to develop a **multiclass classification model** to predict an individual's **obesity risk level**.  

The target variable was a categorical feature representing multiple obesity risk levels (e.g., **Insufficient weight, normal weight, overweight level I,overweight level II, obesity type I, obesity type II, and obesity type III**). The primary goal was to predict the correct obesity class based on an individual’s health, lifestyle, and demographic attributes.  

The evaluation metric for the competition was **accuracy**, which measures the percentage of correctly classified instances. The focus was on building an accurate and interpretable model to achieve competitive results.  

---

## Dataset Description  
The dataset included:  
- **Training Data**: Labeled data with lifestyle, health, and demographic features along with the target obesity risk level.  
- **Testing Data**: Similar data without the target obesity level, used for generating predictions.  
- **Features**: Key features used for prediction included:  
  - **Lifestyle Factors**: Eating habits, frequency of physical activity, and mode of transportation used   
  - **Health Metrics**: Weight, cholesterol levels,Height, Family History with overweight.  
  - **Demographic Features**: Age, gender.  

The dataset presented a **multiclass classification challenge** with varying distributions of obesity risk levels.  

---

## Methodology  

### Exploratory Data Analysis (EDA)  
1. **Feature Exploration**:  
   - Investigated correlations between health metrics (e.g., BMI and obesity level).  
   - Analyzed the distribution of obesity risk levels and identified class imbalance.  

2. **Class Imbalance**:  
   - Certain obesity risk categories had fewer samples, leading to potential bias in predictions. This was mitigated using class weighting techniques.  

3. **Data Preprocessing**:  
   - Scaled numerical features (e.g., BMI, caloric intake) using standardization for compatibility with SVM.  
   - Handled missing values through imputation.  
   - Encoded categorical variables using label encoding.    

---

## Modeling  
To solve the multiclass classification problem, the following models were implemented and evaluated:  

#### 1. Support Vector Machine (SVM)  
- **What It Does**: Maps input features into a high-dimensional space and finds the hyperplane that separates classes with the largest margin.  
- **Strengths**:  
  - Effective for high-dimensional data.  
  - Handles both linear and non-linear relationships when paired with kernels.  
- **Usefulness**:  
  - SVM served as a strong baseline model and was particularly effective when using an RBF kernel to account for non-linear patterns in the dataset.  

#### 2. Random Forest Classifier  
- **What It Does**: Combines predictions from multiple decision trees to enhance accuracy and reduce overfitting.  
- **Strengths**:  
  - Handles mixed data types (numerical and categorical) seamlessly.  
  - Provides robust feature importance metrics for interpretability.  
  - Excels in modeling non-linear relationships and feature interactions.  
- **Usefulness**:  
  - The Random Forest model performed the best in terms of accuracy and generalization, making it the ideal choice for the final submission.  
  - Its feature importance metrics helped identify key predictors of obesity risk, such as caloric intake, BMI, and physical activity frequency.  

#### 3. XGBoost Classifier  
- **What It Does**: Uses gradient boosting to iteratively minimize classification error.  
- **Strengths**:  
  - Highly efficient and powerful for tabular data.  
  - Includes regularization techniques to prevent overfitting.  
- **Usefulness**:  
  - XGBoost provided competitive results and was particularly effective at handling the dataset’s imbalanced class distributions.  

---

### Hyperparameter Tuning  
Hyperparameter optimization was applied to each model to improve performance:  
- **SVM**: Fine-tuned parameters such as kernel type (RBF), regularization strength (`C`), and kernel coefficient (`gamma`).  
- **Random Forest**: Optimized the number of trees, maximum depth, and minimum samples per leaf using grid search.  
- **XGBoost**: Tuned learning rate, maximum depth, and class weights to address class imbalance and improve accuracy.  

The **Random Forest Classifier** emerged as the best-performing model and was selected for final submission based on its superior accuracy and robustness.  

---

## Results  
- **Best Model**: Random Forest achieved the highest accuracy on the validation set, outperforming other models.  
- **Kaggle Submission**: Predictions from the Random Forest model were submitted, achieving competitive results on the leaderboard.  

---

## Key Takeaways  
1. **EDA and Feature Engineering**: Understanding the relationships between health, lifestyle, and demographic features was crucial for improving model performance.  
2. **Random Forest Classifier**: The model’s ability to handle mixed data types and provide interpretable results made it the best choice for this problem.  
3. **Model Selection**: Ensemble models like Random Forest and XGBoost outperformed simpler models like SVM due to their ability to capture non-linear interactions.  

---

## Future Work  
- Incorporate additional external datasets to improve model generalization.  
- Address class imbalance more effectively using oversampling techniques like SMOTE.  
- Experiment with advanced ensemble methods, such as stacking, to combine the strengths of multiple models.  
- Explore deep learning models to capture higher-level feature interactions and patterns.  

---

This project highlights the application of multiclass classification techniques to predict obesity risk levels based on health and lifestyle attributes. If you have any questions or suggestions, feel free to connect!  
