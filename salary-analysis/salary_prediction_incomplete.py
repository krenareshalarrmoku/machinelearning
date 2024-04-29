import pandas as pd
import sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

merged_df = pd.read_csv('../data/classified_job_postings_with_salary.csv')

# Feature Engineering: Handle missing values in 'annual_salary' (fill with -1 for now)
# Explicitly fill missing values in 'annual_salary' with -1
merged_df['annual_salary'] = merged_df['annual_salary'].fillna(-1)
# Create 'has_salary' attribute based on 'annual_salary'
merged_df['has_salary'] = (merged_df['annual_salary'] != -1).astype(int)
# Save the updated DataFrame
merged_df.to_csv('../data/merged_with_has_salary.csv', index=False)

# Load dataset with 'has_salary' attribute
merged_df = pd.read_csv('../data/merged_with_has_salary.csv')

# Identify non-numeric columns
non_numeric_columns = merged_df.select_dtypes(exclude=['number']).columns

# Handle non-numeric columns
for column in non_numeric_columns:
    if merged_df[column].dtype == 'object':
        # Example: Use label encoding for categorical columns
        merged_df[column] = merged_df[column].astype('category').cat.codes
    elif merged_df[column].dtype == 'datetime64':
        # Example: Convert datetime values to numeric representations
        merged_df[column] = merged_df[column].astype(int)

# Filter records with salary information
df_with_salary = merged_df[merged_df['has_salary'] == 1]

# Filter records without salary information
df_without_salary = merged_df[merged_df['has_salary'] == 0]

# Define features and target variable
X = df_with_salary.drop(columns=['annual_salary'])
y = df_with_salary['annual_salary']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict salaries for records without salary information
X_predict = df_without_salary.drop(columns=['annual_salary'])
predicted_salaries = rf_regressor.predict(X_predict)

# Update the merged DataFrame with predicted salaries
merged_df.loc[~merged_df['has_salary'], 'annual_salary'] = predicted_salaries

# Evaluate the model (Optional)
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the updated DataFrame with predicted salaries
merged_df.to_csv('merged_with_predicted_salaries.csv', index=False)