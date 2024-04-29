# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix
import re

# Load datasets
job_postings = pd.read_csv('data/job_postings.csv')
job_summaries = pd.read_csv('data/job_summary.csv')
job_skills = pd.read_csv('data/job_skills.csv')

# Merge datasets
merged_data = job_postings.merge(job_summaries, on='job_link', how='left')
merged_data = merged_data.merge(job_skills, on='job_link', how='left')

# Drop irrelevant columns and missing values
merged_data.drop(['job_link', 'last_processed_time', 'last_status', 'got_summary', 'got_ner', 'is_being_worked'], axis=1, inplace=True)
merged_data.dropna(inplace=True)

# Extract individual skills
merged_data['job_skills'] = merged_data['job_skills'].apply(lambda x: re.findall(r'\w+', x))

# Convert skills list to string
merged_data['job_skills'] = merged_data['job_skills'].apply(lambda x: ' '.join(x))


# Define features and target
X = merged_data['job_skills']  # Using job skills as features
y = merged_data['job_level']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize job skills using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = lr_model.predict(X_test_tfidf)

# Evaluate model using precision score and confusion matrix
precision = precision_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Precision Score:", precision)
print("Confusion Matrix:")
print(conf_matrix)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=lr_model.classes_, yticklabels=lr_model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Bar plot for comparing actual vs. predicted job levels
plt.figure(figsize=(10, 6))
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
comparison_df['Comparison'] = comparison_df['Actual'] == comparison_df['Predicted']
comparison_df['Comparison'] = comparison_df['Comparison'].replace({True: 'Correct', False: 'Incorrect'})
comparison_df_grouped = comparison_df.groupby(['Actual', 'Comparison']).size().unstack(fill_value=0)
comparison_df_grouped.plot(kind='bar', stacked=True)
plt.xlabel('Job Levels')
plt.ylabel('Count')
plt.title('Actual vs. Predicted Job Levels')
plt.legend(title='Prediction Accuracy')
plt.xticks(rotation=45)
plt.show()