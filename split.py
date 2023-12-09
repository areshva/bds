import pandas as pd

# Path of your filtered CSV
input_file_path = '02909_c.csv'

# Read the filtered csv file into a pandas DataFrame
df = pd.read_csv(input_file_path)

# # Define the names of the output files
# output_files = [
#     "loss_urgent.csv",
#     "loss_noturgent.csv",
#     "gain_urgent.csv",
#     "gain_noturgent.csv",
#     "neutral_urgent.csv",
#     "neutral_noturgent.csv"
# ]

# # Check if the DataFrame has at least 1800 rows
# if len(df) < 1800:
#     raise ValueError("The input file does not have enough rows (1800 required)")

# # Split the DataFrame into chunks and write to separate CSVs
# for i, file_name in enumerate(output_files):
#     start_index = i * 300  # starting index of the chunk
#     end_index = (i + 1) * 300  # ending index of the chunk
#     chunk = df[start_index:end_index]  # get the chunk
#     chunk.to_csv(file_name, index=False)  # write the chunk to CSV
#     print(f"{file_name} has been created with rows {start_index} to {end_index}")

# print("All files created successfully.")


# Check if the DataFrame has at least 300 rows
if len(df) < 300:
    raise ValueError("The input file does not have enough rows (at least 300 required)")

# Extract the last 300 rows from the DataFrame
last_300_rows = df.tail(300)

# Define the name of the output file
output_file = "loss-nu2.csv"

# Write the last 300 rows to the output CSV file
last_300_rows.to_csv(output_file, index=False)

print(f"{output_file} has been created with the last 300 rows.")
