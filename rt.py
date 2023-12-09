import pandas as pd
from urllib.parse import urlparse, parse_qs

# Load the CSV with click data
click_data_path = 'clicks.csv'
click_df = pd.read_csv(click_data_path, parse_dates=['timestamp'])

# Extract utm_source and utm_medium from the URL
def extract_utm_parameters(url):
    params = parse_qs(urlparse(url).query)
    return params.get('utm_source', [None])[0], params.get('utm_medium', [None])[0]

click_df['utm_source'], click_df['utm_medium'] = zip(*click_df['url'].apply(extract_utm_parameters))

# Define the timestamps when the links were sent out
sent_times = {
    'gain-urgent': pd.Timestamp('2023-10-25 15:16:00'),
    'gain-noturgent': pd.Timestamp('2023-10-25 17:57:00'),
    'neutral-urgent': pd.Timestamp('2023-10-18 18:30:00'),
    'neutral-noturgent': pd.Timestamp('2023-10-25 17:33:00'),
    'loss-urgent': pd.Timestamp('2023-10-25 15:16:00')
}

# Filter records and calculate response times
response_times = {}
total_clicks = {}
for key, sent_time in sent_times.items():
    # Calculate start and end times for each group
    start_time = sent_time
    end_time = sent_time + pd.Timedelta(hours=24)
    
    source, medium = key.split('-')
    filtered = click_df[(click_df['utm_source'] == source) & 
                        (click_df['utm_medium'] == medium) & 
                        (click_df['timestamp'] >= start_time) & 
                        (click_df['timestamp'] <= end_time)]
    
    time_diffs = (filtered['timestamp'] - sent_time).dt.total_seconds() / 60  # in minutes
    response_times[key] = time_diffs
    total_clicks[key] = len(time_diffs)

# Analyze the results
urgent_gain_response_time = response_times['gain-urgent'].mean()
noturgent_gain_response_time = response_times['gain-noturgent'].mean()
urgent_neutral_response_time = response_times['neutral-urgent'].mean()
noturgent_neutral_response_time = response_times['neutral-noturgent'].mean()
urgent_loss_response_time = response_times['neutral-noturgent'].mean()



print(f"Average response time for urgent-gain: {urgent_gain_response_time:.2f} minutes (Total Clicks: {total_clicks['gain-urgent']})")
print(f"Average response time for noturgent-gain: {noturgent_gain_response_time:.2f} minutes (Total Clicks: {total_clicks['gain-noturgent']})")
print(f"Average response time for urgent-neutral: {urgent_neutral_response_time:.2f} minutes (Total Clicks: {total_clicks['neutral-urgent']})")
print(f"Average response time for noturgent-neutral: {noturgent_neutral_response_time:.2f} minutes (Total Clicks: {total_clicks['neutral-noturgent']})")
print(f"Average response time for urgent-loss: {urgent_loss_response_time:.2f} minutes (Total Clicks: {total_clicks['loss-urgent']})")

results = {}
for key in response_times:
    results[key] = {
        'mean': response_times[key].mean(),
        'median': response_times[key].median(),
        'total_clicks': total_clicks[key]
    }

# Print the results
for key, values in results.items():
    print(f"Results for {key}:")
    print(f"  Average response time (mean): {values['mean']:.2f} minutes")
    print(f"  Median response time: {values['median']:.2f} minutes")
    print(f"  Total Clicks: {values['total_clicks']}")
    print("-" * 50)


# from scipy.stats import mannwhitneyu

# # Example: Compare response times between 'urgent-gain' and 'noturgent-gain'
# group1 = response_times['neutral-urgent']
# group2 = response_times['neutral-noturgent']

# # Mann-Whitney U test
# stat, p = mannwhitneyu(group1, group2, alternative='two-sided')

# # Print the results
# if p < 0.05:
#     print(f"The difference in response times between 'urgent-neutral' and 'noturgent-neutral' is statistically significant (p={p:.5f}).")
# else:
#     print(f"There's no statistically significant difference in response times between 'urgent-neutral' and 'noturgent-neutral' (p={p:.5f}).")
