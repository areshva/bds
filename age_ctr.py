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

print(ctr_age_group)

import statsmodels.api as sm
df_filtered = df_filtered.dropna(subset=['est_age'])

# Prepare the data for logistic regression
X = df_filtered['est_age']
X = sm.add_constant(X)  # adding a constant
y = df_filtered['clicked']

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Print the model summary
#print(result.summary())

#Between Conditions effects:
#CTR
pivot_table = df_filtered.pivot_table(index='age_group', columns='condition', values='clicked', aggfunc='sum').fillna(0)

# Get the counts of each age group for each condition
age_group_counts = df_filtered.groupby(['age_group', 'condition']).size().unstack().fillna(0)

# Calculate the CTR
ctr_table = pivot_table / age_group_counts

#print(ctr_table)


ctr_table.to_csv('ctr_table.csv', index=True)

import pandas as pd

# ... Your existing code...

ctr_table.to_csv('ctr_table.csv', index=True)

# Reading in the CSV data
df_ctr = pd.read_csv('ctr_table.csv')

# Define urgent and not urgent conditions
urgent_conditions = ['gain_urgent', 'loss_urgent', 'neutral_urgent']
not_urgent_conditions = ['gain_noturgent', 'loss_noturgent', 'neutral_noturgent']

# Compute average CTR for urgent and not urgent conditions
df_ctr['avg_urgent'] = df_ctr[urgent_conditions].mean(axis=1)
df_ctr['avg_not_urgent'] = df_ctr[not_urgent_conditions].mean(axis=1)

# Print results
for index, row in df_ctr.iterrows():
    age_group = row['age_group']
    avg_urgent = row['avg_urgent']
    avg_not_urgent = row['avg_not_urgent']
    
    print(f"{age_group} Age Group:")
    print(f"Average CTR for urgent conditions is around {avg_urgent:.3f} or {avg_urgent*100:.1f}%.")
    print(f"Average CTR for not urgent conditions is around {avg_not_urgent:.3f} or {avg_not_urgent*100:.1f}%.")
    print("-" * 50)



#ABSOLUTE NUMBERS
