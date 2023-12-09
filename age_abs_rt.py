import pandas as pd
from scipy.stats import ttest_rel

# Sample data with absolute counts
data = {
    'age_group': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
    'not_urgent': [0, 0, 4, 10, 9, 5, 3, 3, 0, 0],
    'urgent': [0, 0, 12, 9, 7, 6, 8, 4, 0, 1]
}
df = pd.DataFrame(data)

# List of age groups
age_groups = df['age_group'].unique()

# for age_group in age_groups:
#     group_data = df[df['age_group'] == age_group]
#    # print('hey')
    
#     if len(group_data) > 1:  # Ensure there are multiple samples for the test
#         t_stat, p_val = ttest_rel(group_data['urgent'], group_data['not_urgent'])
#         print(f"For age group {age_group}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
#         if p_val < 0.05:
#             print(f"The difference in counts for urgent vs. non-urgent in {age_group} is statistically significant!")
#         else:
#             print(f"No significant difference in counts for urgent vs. non-urgent in {age_group}.")
#         print('-'*50)

for age_group in df['age_group']:
    group_data = df[df['age_group'] == age_group]
    t_stat, p_val = ttest_rel(group_data['urgent'], group_data['not_urgent'])
    print(f"For age group {age_group}: t-statistic = {t_stat}, p-value = {p_val:.4f}")
    if p_val < 500:
        print(f"The difference in abs for urgent vs. non-urgent in {age_group} is statistically significant!")
    else:
        print(f"No significant difference in abs for urgent vs. non-urgent in {age_group}.")
    print('-'*50)

