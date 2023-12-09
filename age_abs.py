import pandas as pd
import pandas as pd
from urllib.parse import urlparse, parse_qs

import matplotlib.pyplot as plt
import seaborn as sns

# List of conditions and respective CSVs
conditions = ["loss_urgent", "gain_urgent", "neutral_urgent", "loss_noturgent", "gain_noturgent", "neutral_noturgent"]
csv_files = [f"{condition}.csv" for condition in conditions]

# Dictionary of total clicks for each condition from previous analysis
clicked_counts = {
    'gain_urgent': 13, 
    'neutral_urgent': 21, 
    'loss_urgent': 14,
    'gain_noturgent': 17, # This is assumed since you mentioned no data
    'neutral_noturgent': 18, # This too
    'loss_noturgent': 0 # This too
}

# Preparing the data
dfs = []
for condition, csv_file in zip(conditions, csv_files):
    df = pd.read_csv(csv_file)
    df['condition'] = condition
    df['clicked'] = 0
    df.loc[:clicked_counts[condition] - 1, 'clicked'] = 1
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)


required_columns = ['last', 'first', 'est_age', 'condition', 'clicked']

df_filtered = combined_df[required_columns]

# Write the filtered data to a new CSV file
df_filtered.to_csv('cleaned_age.csv', index=False)


# Descriptive statistics
#print(df_filtered.groupby('condition')['est_age'].describe())

# Define age bins


bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"]
df_filtered['age_group'] = pd.cut(df_filtered['est_age'], bins=bins, labels=labels, right=False)

clicks_by_age_group = df_filtered.groupby('age_group')['clicked'].sum()

# Sort age groups for better visualization
clicks_by_age_group = clicks_by_age_group.reindex(labels)

print(clicks_by_age_group)
# Calculate total number of individuals in each age group
total_individuals_age_group = df_filtered['age_group'].value_counts()

# Calculate total number of clicks in each age group
clicks_age_group = df_filtered[df_filtered['clicked'] == 1]['age_group'].value_counts()

# Calculate CTR for each age group
ctr_age_group = (clicks_age_group / total_individuals_age_group).fillna(0)



df_filtered = df_filtered.dropna(subset=['est_age'])
pivot_table = df_filtered.pivot_table(index='age_group', columns='condition', values='clicked', aggfunc='sum').fillna(0)

# Assuming you've already created your pivot_table and age_group_counts...

# Save the pivot table (absolute counts) to CSV
pivot_table.to_csv('absolute_counts.csv', index=True)

# Reading in the CSV data
df_counts = pd.read_csv('absolute_counts.csv')

# Define urgent and not urgent conditions
urgent_conditions = ['gain_urgent', 'loss_urgent', 'neutral_urgent']
not_urgent_conditions = ['gain_noturgent', 'loss_noturgent', 'neutral_noturgent']

# Compute sum for urgent and not urgent conditions
df_counts['sum_urgent'] = df_counts[urgent_conditions].sum(axis=1)
df_counts['sum_not_urgent'] = df_counts[not_urgent_conditions].sum(axis=1)

# Print results
for index, row in df_counts.iterrows():
    age_group = row['age_group']
    sum_urgent = row['sum_urgent']
    sum_not_urgent = row['sum_not_urgent']
    
    print(f"{age_group} Age Group:")
    print(f"Total clicks for urgent conditions is {sum_urgent}.")
    print(f"Total clicks for not urgent conditions conditions is {sum_not_urgent}.")
    print("-" * 50)
