import pandas as pd
import re
import os

# Sample data loading steps. Adjust paths and formats according to your actual data sources.
# Assuming the datasets are provided in a similar format as the input given.

for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

job_summary_df = pd.read_csv('data/job_summary.csv')
job_postings_df = pd.read_csv('data/job_postings.csv')
job_skills_df = pd.read_csv('data/job_skills.csv')

# Step 1: Filter out records where 'job_link' contains "request is blocked".
job_summary_df = job_summary_df[~job_summary_df['job_link'].str.contains("request is blocked", na=False)]
job_postings_df = job_postings_df[~job_postings_df['job_link'].str.contains("request is blocked", na=False)]
job_skills_df = job_skills_df[~job_skills_df['job_link'].str.contains("request is blocked", na=False)]

# Step 2: Remove duplicate entries based on 'job_link'.
job_summary_df.drop_duplicates(subset=['job_link'], inplace=True)
job_postings_df.drop_duplicates(subset=['job_link'], inplace=True)
job_skills_df.drop_duplicates(subset=['job_link'], inplace=True)

# Step 3: Normalize text fields (e.g., job_summary).
job_summary_df['job_summary'] = job_summary_df['job_summary'].apply(
    lambda x: re.sub('[^A-Za-z0-9 ]+', '', str(x).lower()))


# Step 4: Extract and standardize 'job_location' into 'City' and 'State/Country'.
def standardize_location(location):
    if pd.isnull(location):
        return "Unknown", "Unknown"
    parts = location.split(", ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return location.strip(), "Unknown"


job_postings_df['City'], job_postings_df['State/Country'] = zip(
    *job_postings_df['job_location'].apply(standardize_location))

# Step 5: Merge job postings with job skills on 'job_link'.
merged_df = pd.merge(job_postings_df, job_skills_df, on='job_link', how='left')

merged_df = pd.merge(merged_df, job_summary_df, on='job_link', how='left')

# Optionally, save the cleaned and merged dataframe to a new CSV file.
merged_df.to_csv('../data/cleaned_merged_job_data.csv', index=False)

print("Data cleaning and merging completed. Saved to '../data/cleaned_merged_job_data.csv'.")
