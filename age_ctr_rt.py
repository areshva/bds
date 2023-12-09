import pandas as pd
from scipy.stats import ttest_rel

# Sample data
data = {
    'age_group': ['0-9', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
    'not_urgent': [0.000, 0.032, 0.043, 0.064, 0.030, 0.026, 0.045, 0.000, 0.000],
    'urgent': [0.000, 0.092, 0.036, 0.042, 0.049, 0.058, 0.071, 0.000, 0.333]
}
df = pd.DataFrame(data)

# Conduct paired t-test for each age group
for age_group in df['age_group']:
    group_data = df[df['age_group'] == age_group]
    t_stat, p_val = ttest_rel(group_data['urgent'], group_data['not_urgent'])
    print(f"For age group {age_group}: t-statistic = {t_stat}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print(f"The difference in CTR for urgent vs. non-urgent in {age_group} is statistically significant!")
    else:
        print(f"No significant difference in CTR for urgent vs. non-urgent in {age_group}.")
    print('-'*50)
