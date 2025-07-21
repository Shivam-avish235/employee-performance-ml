# Employee Performance Prediction using Machine Learning

This project aims to predict the **actual productivity** of garment factory workers based on various features like SMV, overtime, idle time, style changes, and more. Using machine learning algorithms (Linear Regression, Random Forest, and XGBoost), the model provides a reliable prediction of productivity.

## Dataset

- **Source**: `garments_worker_productivity.csv`
- **Total Records**: 1197
- **Features Used**:
  - Quarter
  - Department
  - Day
  - Team
  - Targeted Productivity
  - SMV
  - WIP
  - Over Time
  - Incentive
  - Idle Time
  - Idle Men
  - No. of Style Change
  - No. of Workers
  - Month

## Project Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical features
   - Extracting month from date
2. **Model Building**
   - Linear Regression
   - Random Forest
   - XGBoost
3. **Evaluation**
   - MAE, MSE, R² Score
4. **Best Model Saving**
   - The best performing model (XGBoost) is saved as `best_model.pkl`
5. **Flask Web App**
   - User can input details through a form
   - Predicted productivity is displayed in real-time
6. **Deployment Ready**
   - Can be deployed on local server or platforms like Render, Heroku, or Replit

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Flask
- HTML, CSS (Bootstrap)
- Jupyter Notebook


