import pandas as pd
from urllib.parse import urlparse, parse_qs

# 1. Load the CSV and Extract UTM Parameters
click_data_path = 'clicks.csv'
click_df = pd.read_csv(click_data_path, parse_dates=['timestamp'])

def extract_utm_parameters(url):
    params = parse_qs(urlparse(url).query)
    return params.get('utm_source', [None])[0], params.get('utm_medium', [None])[0]

click_df['utm_source'], click_df['utm_medium'] = zip(*click_df['url'].apply(extract_utm_parameters))

# 2. Define the timestamps when the links were sent out
sent_times = {
    'gain-urgent': pd.Timestamp('2023-10-25 15:16:00'),
    'gain-noturgent': pd.Timestamp('2023-10-25 17:57:00'),
    'neutral-urgent': pd.Timestamp('2023-10-18 18:30:00'),
    'neutral-noturgent': pd.Timestamp('2023-10-25 17:33:00'),
    'loss-urgent': pd.Timestamp('2023-10-25 15:16:00'),
   # 'loss-noturgent': pd.Timestamp('2023-11-25 16:13:00')
}


# 3. Filter records and calculate response times
response_times = {}
total_clicks = {}
for key, sent_time in sent_times.items():
    source, medium = key.split('-')
    filtered = click_df[(click_df['utm_source'] == source) & 
                        (click_df['utm_medium'] == medium) &
                        (click_df['timestamp'] >= sent_time) &
                        (click_df['timestamp'] <= sent_time + pd.Timedelta(hours=24))]
    
    time_diffs = (filtered['timestamp'] - sent_time).dt.total_seconds() / 60  # in minutes
    response_times[key] = time_diffs
    total_clicks[key] = len(time_diffs)

print(total_clicks)

# 4. Calculate descriptive statistics and click rates for each condition
stats_data = {
    'condition': [],
    'mean_time': [],
    'median_time': [],
    'std_dev_time': [],
    'click_rate': []
}

total_links_sent = {
    'gain-urgent': 232, 
    'neutral-urgent': 213,  
    'loss-urgent': 202  
}
total_sent = {
    'gain-urgent': 232,
    'neutral-urgent': 213,
    'loss-urgent': 202,
    'gain-noturgent': 201,
    'neutral-noturgent': 211,
   # 'loss-noturgent': 204,

}

for condition, total_sent in total_sent.items():
    stats_data['condition'].append(condition)
    stats_data['mean_time'].append(response_times[condition].mean())
    stats_data['median_time'].append(response_times[condition].median())
    stats_data['std_dev_time'].append(response_times[condition].std())
    stats_data['click_rate'].append(total_clicks[condition] / total_sent)

stats_df = pd.DataFrame(stats_data)
print(stats_df)

total_sent = {
    'gain-urgent': 232,
    'neutral-urgent': 213,
    'loss-urgent': 202,
    'gain-noturgent': 201,
    'neutral-noturgent': 211,

}

clicked_counts = {
    'gain-urgent': total_clicks.get('gain-urgent', 0),
    'neutral-urgent': total_clicks.get('neutral-urgent', 0),
    'loss-urgent': total_clicks.get('loss-urgent', 0),
    'gain-noturgent': total_clicks.get('gain-noturgent', 0),
    'neutral-noturgent': total_clicks.get('neutral-noturgent', 0)

}

not_clicked_counts = {key: total_sent[key] - clicked_counts[key] for key in clicked_counts.keys()}

contingency_table = pd.DataFrame({
    'Condition': ['gain-urgent', 'neutral-urgent', 'loss-urgent'],
    'Clicked': [clicked_counts['gain-urgent'], clicked_counts['neutral-urgent'], clicked_counts['loss-urgent']],
    'Not-Clicked': [not_clicked_counts['gain-urgent'], not_clicked_counts['neutral-urgent'], not_clicked_counts['loss-urgent']]
})

print(contingency_table)

contingency_table.to_csv('contingency_table_h1.csv', index=False)


from scipy.stats import chi2_contingency

# Using the Clicked and Not-Clicked columns for the Chi-squared test
observed = contingency_table[['Clicked', 'Not-Clicked']].values

chi2, p, _, _ = chi2_contingency(observed)

print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")
