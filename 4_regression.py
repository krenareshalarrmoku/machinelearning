import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load datasets
job_postings = pd.read_csv('data/job_postings.csv')
job_skills = pd.read_csv('data/job_skills.csv')  # Assuming a typo in the filename
job_summary = pd.read_csv('data/job_summary.csv')

# Merge datasets on 'job_link'
merged_data = job_postings.merge(job_skills, on='job_link').merge(job_summary, on='job_link')
merged_data.fillna({'job_skills': ''}, inplace=True)
# Preprocess data
# Assuming 'job_level' is the target and mapping it to a numerical scale
job_level_mapping = {
    'Associate': 1,
    'Mid senior': 2,
    'Senior': 3,
    'Lead': 4,
    'Principal': 5
}
merged_data['job_level_num'] = merged_data['job_level'].map(job_level_mapping)

# Feature engineering for text data using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
skills_tfidf = tfidf.fit_transform(merged_data['job_skills']).toarray()
summary_tfidf = tfidf.fit_transform(merged_data['job_summary']).toarray()

# Combining all features
X_all_features = pd.concat([
    pd.DataFrame(skills_tfidf),
    pd.DataFrame(summary_tfidf),
], axis=1)

y = merged_data['job_level_num']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all_features, y, test_size=0.4, random_state=42)



# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# For regression, we'll focus on MSE and use negative MSE as the scoring method because cross_val_score
# interprets higher scores as better, whereas we want to minimize MSE.
pipeline = make_pipeline(StandardScaler(), LinearRegression())

# Performing 5-Fold Cross-Validation and using 'neg_mean_squared_error' as scoring
scores = cross_val_score(pipeline, X_all_features, y, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive MSE scores
mse_scores = -scores

print(f'MSE scores for the 5 folds: {mse_scores}')
print(f'Average MSE: {mse_scores.mean()}')

import matplotlib.pyplot as plt

# Plotting Predicted vs. Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Job Levels')
plt.ylabel('Predicted Job Levels')
plt.title('Predicted vs. Actual Job Levels')
plt.show()


# Calculate residuals
residuals = y_test - y_pred

# Plotting Residuals
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, color='green', alpha=0.5)
plt.xlabel('Actual Job Levels')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.title('Residual Plot')
plt.show()


import matplotlib.pyplot as plt

# Plotting MSE scores for the 5 folds
plt.figure(figsize=(8, 6))

plt.bar(range(1, 6), mse_scores)
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title('Mean Squared Error for Cross-Validation Folds')
plt.show()
