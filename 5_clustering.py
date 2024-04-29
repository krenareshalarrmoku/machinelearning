import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

# Load datasets from CSV files
job_postings_df = pd.read_csv('data/job_postings.csv')
job_summary_df = pd.read_csv('data/job_summary.csv')
job_skills_df = pd.read_csv('data/job_skills.csv')

# Merge datasets on 'job_link'
merged_df = job_postings_df.merge(job_summary_df, on='job_link', how='left')\
                           .merge(job_skills_df, on='job_link', how='left')

# Simple preprocessing: Lowercasing and replacing commas with spaces in 'job_skills'
merged_df['job_skills'] = merged_df['job_skills'].str.lower().replace(',', ' ', regex=False)

# Fill any missing values in 'job_skills' with an empty string
merged_df['job_skills'] = merged_df['job_skills'].fillna('')

# Vectorize the 'job_skills' text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(merged_df['job_skills'])

# Normalize the TF-IDF features
X_normalized = normalize(X_tfidf, norm='l1', axis=1)

# Cluster the job postings using K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_normalized)

# Add the cluster assignments to your DataFrame
merged_df['cluster'] = clusters

# Perform TruncatedSVD for visualization
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_normalized)

# Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('Cluster Visualization using TruncatedSVD')
plt.xlabel('Technical')
plt.ylabel('Non-technical')
plt.colorbar(label='Cluster')
plt.show()

# At this point, you can export merged_df to a new CSV for further analysis or inspection if needed
merged_df.to_csv('data/classified_job_postings_cluster.csv', index=False)

# Example: Display the distribution of job postings across the generated clusters
print(merged_df['cluster'].value_counts())
