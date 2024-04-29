import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")


def extract_salary_with_nlp(description):
    doc = nlp(description)
    salaries = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
    return salaries if salaries else "Salary not specified"


def salary_to_int(salary_list):
    hours_per_week = 40
    weeks_per_year = 52
    results = []

    for salary_str in salary_list:
        salary_str = salary_str.replace(',', '').lower()
        if 'k' in salary_str:
            salary_str = salary_str.replace('k', '000')
            if '-' in salary_str:
                try:
                    low, high = [int(s.replace('$', '')) for s in salary_str.split('-')]
                    average_salary = (low + high) // 2
                    if average_salary < 1000000:
                        results.append(average_salary)
                except:
                    continue
            else:
                try:
                    if int(salary_str.replace('$', '')) < 1000000:
                        results.append(int(salary_str.replace('$', '')))
                except:
                    continue

        elif '/hr' in salary_str:
            try:
                hourly_rate = int(salary_str.replace('$', '').split('/')[0])
            except:
                continue

            if hourly_rate > 100:
                results.append(hourly_rate)
            else:
                annual_salary = hourly_rate * hours_per_week * weeks_per_year
                if annual_salary < 1000000:
                    results.append(annual_salary)

        elif any(char.isdigit() for char in salary_str):  # Check if there's any digit
            try:
                if int(salary_str) < 100:
                    results.append(int(salary_str) * hours_per_week * weeks_per_year)
                else:
                    results.append(int(salary_str))
            except:
                continue

            else:
                try:
                    results.append(int(salary_str.replace('$', '')))
                except ValueError:
                    results.append(0)

            return sum(results) // len(results) if results else 0


def mark_positions_by_keyword(df, column_name, keywords, new_column_names):
    for i in range(len(keywords)):
        df[new_column_names[i]] = df[column_name].str.contains(keywords[i], case=False, na=False)

    return df


def extract_state(location):
    try:
        parts = location.split(', ')
        if len(parts) >= 2:
            return parts[1]
        else:
            return None
    except:
        return None


def extract_city(location):
    try:
        parts = location.split(', ')
        if len(parts) >= 2:
            return parts[1]
        else:
            return None
    except:
        return None
