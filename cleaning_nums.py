import pandas as pd

# Specify the path of your input file
input_file_path = '02909_c.csv'  # replace with your input file name

# Specify the path of your output file
output_file_path = 'numbers_output.csv'  # replace with your desired output file name

# Read the csv file into a pandas DataFrame
df = pd.read_csv(input_file_path)

# Check if the required columns exist in the DataFrame
required_columns = ['last', 'first', 'cell']
if not all(column in df.columns for column in required_columns):
    raise ValueError("Not all required columns are present in the input file")

# Select only the 'last', 'first', and 'cell' columns
df_filtered = df[required_columns]

# Write the filtered data to a new CSV file
df_filtered.to_csv(output_file_path, index=False)

print(f"Filtered CSV created successfully as {output_file_path}")
