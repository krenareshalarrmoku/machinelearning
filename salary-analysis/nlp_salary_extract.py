import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from helpers import salary_to_int, extract_salary_with_nlp, extract_state, mark_positions_by_keyword


df = pd.read_csv("../data/classified_job_postings.csv")
filtered_df = df[df['category'] == 'technical']
filtered_df = mark_positions_by_keyword(df, 'job_title', ['Staff', 'Manager', 'Senior', 'Director', 'Software Engineer',
                                                 'Data Engineer', 'Machine Learning Engineer', 'Researcher'],
                               ['is_staff', 'is_manager', 'is_senior', 'is_director', 'is_swe', 'is_data_eng',
                                'is_mle', 'is_researcher'])

filtered_df['state'] = df['job_location'].apply(extract_state)

from tqdm import tqdm

tqdm.pandas(desc="NLP extracting salaries")

filtered_df['extracted_salaries'] = filtered_df['job_summary'].progress_apply(extract_salary_with_nlp)
filtered_df['annual_salary'] = filtered_df['extracted_salaries'].progress_apply(salary_to_int)
df_sh = filtered_df[filtered_df['annual_salary'] != 0.0]

df_sh.to_csv('../data/classified_job_postings_with_salary.csv', index=False)

