import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load datasets
job_postings = pd.read_csv('data/job_postings.csv')
job_skills = pd.read_csv('data/job_skills.csv')  # Assuming a typo in the filename
job_summary = pd.read_csv('data/job_summary.csv')

# Merge datasets on 'job_link'
merged_data = job_postings.merge(job_skills, on='job_link').merge(job_summary, on='job_link')
merged_data.fillna({'job_skills': ''}, inplace=True)
# Define senior level jobs as 1 and others as 0
merged_data['is_senior'] = merged_data['job_title'].apply(lambda x: 1 if x in ['Mid senior', 'Lead'] else 0)

# Feature engineering for text data using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
skills_tfidf = tfidf.fit_transform(merged_data['job_skills']).toarray()
summary_tfidf = tfidf.fit_transform(merged_data['job_summary']).toarray()

# Combining all features
X_all_features = pd.concat([
    pd.DataFrame(skills_tfidf),
    pd.DataFrame(summary_tfidf),
], axis=1)

y = merged_data['is_senior']

print(X_all_features,y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all_features, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000) # Increasing max_iter for convergence
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)