import pandas as pd

# Sample data (you would use your actual data)
total_sent = {'gain-urgent': 232, 'gain-noturgent': 201, 'neutral-urgent': 213, 'neutral-noturgent': 211, 'loss-urgent': 202, 'loss-noturgent': 204}
total_clicks = {'gain-urgent': 13, 'gain-noturgent': 17, 'neutral-urgent': 21, 'neutral-noturgent': 18, 'loss-urgent': 14, 'loss-noturgent': 18}

# Reconstruct the dataset
data = []
for condition, clicks in total_clicks.items():
    # Add clicked entries
    for _ in range(clicks):
        data.append([condition, 1])  # 1 for clicked
    # Add not clicked entries
    not_clicked = total_sent[condition] - clicks
    for _ in range(not_clicked):
        data.append([condition, 0])  # 0 for not clicked

# Create DataFrame
df = pd.DataFrame(data, columns=['condition', 'clicked'])

# Split condition into framing and urgency
split_columns = df['condition'].str.split('-', expand=True)
df['framing'] = split_columns[0]
df['urgency'] = split_columns[1]

# One-hot encode the categorical variables
df = pd.get_dummies(df, columns=['framing', 'urgency'])
output_file_path = 'log_reg.csv'
df.to_csv(output_file_path, index=False)
# Resulting DataFrame
print(df.head())
