# Regression Modeling for Predicting Used Car Prices  

## Project Overview  
This project was developed as part of the Kaggle competition. The objective of the competition was to build a predictive model to accurately forecast the **price of used cars** based on various attributes.  

The dataset included information about car features such as mileage, engine size, fuel type, manufacturing year, and more. The task was to leverage these features to predict the target variable, which is the car's price.  

The performance metric for evaluation was the **Root Mean Squared Error (RMSE)**, which measures the deviation between predicted and actual car prices, with a focus on penalizing larger prediction errors. The goal was to minimize RMSE to achieve higher accuracy and rank well on the competition leaderboard.  

---

## Dataset Description  
The dataset consisted of:  
- **Training Data**: Historical data containing features like `year`, `engine_size`, `mileage`, and car attributes such as `fuel_type`, `brand`, and `transmission`.  
- **Testing Data**: Similar data without the target variable (`price`), where predictions were to be generated.  
- **Features**: A mix of numerical and categorical variables. Key features included:  
  - `mileage`, `engine_size`, `year` (numerical features).  
  - `fuel_type`, `transmission`, `brand` (categorical features).  

The challenge was to handle this mixed-type data effectively while capturing complex interactions between the features and the target variable.  

---

## Methodology  

### Exploratory Data Analysis (EDA)  
1. **Descriptive Statistics**:  
   - Analyzed the distribution of car prices to identify outliers and skewness.  
   - Explored key numerical features like `mileage` and `engine_size` to understand their distributions and trends.  

2. **Categorical Analysis**:  
   - Investigated the relationship between categorical variables (`fuel_type`, `transmission`, `brand`) and car prices.  
   - Encoded categorical features using one-hot encoding for certain models and target encoding for others.  

3. **Feature Relationships**:  
   - Used correlation analysi sto assess how numerical features as well categorical features are correlated with car prices. 

4. **Data Cleaning**:  
   - Imputed missing values using mean/mode strategies or grouped statistics.  
   - Scaled numerical variables where required, especially for models sensitive to feature magnitude (e.g., Linear Regression).  

---

## Modeling  
To accurately predict used car prices, I experimented with the following models:  

#### 1. Linear Regression  
- **What It Does**: Establishes a linear relationship between features and the target variable.  
- **Strengths**:  
  - Easy to interpret and implement.  
  - Effective for baseline performance and identifying linear feature relationships.  
- **Usefulness**:  
  - Served as a strong baseline model to evaluate the dataset's linear patterns.  

#### 2. Random Forest Regressor  
- **What It Does**: An ensemble learning method that combines predictions from multiple decision trees to reduce overfitting and improve accuracy.  
- **Strengths**:  
  - Handles categorical and numerical variables effectively.  
  - Can model non-linear relationships and interactions between features.  
  - Provides feature importance scores for better feature selection.  
- **Usefulness**:  
  - Performed well on this dataset by capturing complex relationships between car attributes and prices.  

#### 3. XGBoost Regressor  
- **What It Does**: A powerful gradient boosting framework that builds models iteratively to correct errors from previous iterations.  
- **Strengths**:  
  - Highly efficient and robust, especially for structured/tabular data.  
  - Supports regularization to reduce overfitting.  
  - Offers advanced interpretability through SHAP values.  
- **Usefulness**:  
  - Captured subtle patterns in the data and improved predictions compared to Random Forest after hyperparameter tuning.  

#### 4. LightGBM Regressor (LGBM)  
- **What It Does**: A gradient boosting framework optimized for speed and memory usage, focusing on leaf-wise growth for better accuracy.  
- **Strengths**:  
  - Faster training and prediction compared to XGBoost.  
  - Handles large datasets efficiently and supports categorical features natively.  
  - Provides excellent feature importance scores for interpretation.  
- **Usefulness**:  
  - Outperformed all other models on this dataset, achieving the lowest RMSE. Its speed and accuracy made it the best choice for this competition.  

---

### Hyperparameter Tuning  
Hyperparameter optimization was performed for the ensemble models (Random Forest, XGBoost, and LightGBM) to improve performance:  
- **Random Forest**: Number of estimators, maximum depth, and minimum samples per leaf were fine-tuned.  
- **XGBoost**: Learning rate, maximum depth, number of boosting rounds, and regularization terms were optimized.  
- **LightGBM**: Parameters like learning rate, number of leaves, and boosting type were tuned for the best results.  

The **LightGBM Regressor** emerged as the top-performing model during validation and was selected for submission.  

---

## Results  
- **Best Model**: The LightGBM Regressor achieved the lowest RMSE on the validation set, outperforming other models.  
- **Kaggle Submission**: The predictions generated by the LightGBM model ranked competitively on the leaderboard.  

---

## Key Takeaways  
1. **EDA**: Understanding feature relationships and performing robust preprocessing were critical for improving model performance.  
2. **Model Selection**: Ensemble models like Random Forest, XGBoost, and LightGBM performed significantly better than simpler models like Linear Regression, highlighting the importance of capturing non-linear relationships.  
3. **LightGBM**: The model’s ability to handle mixed data types, efficiency, and high predictive accuracy made it the best choice for this problem.  

---

## Future Work  
- Investigate feature interactions more deeply using advanced techniques like polynomial features or interaction terms.  
- Explore ensemble stacking to combine the strengths of multiple models for further performance gains.  
- Consider deep learning approaches like neural networks for potential improvements, especially with additional feature engineering.  

---

This project demonstrates the application of regression modeling techniques to predict used car prices accurately. If you have any questions or would like to collaborate, feel free to reach out!  
