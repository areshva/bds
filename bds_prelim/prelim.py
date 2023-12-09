import csv

# Define the path to your CSV file. Adjust this as necessary.
file_path = "bds_prelim/gain_noturgent.csv"

def count_delivered_status(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        # Count the number of entries with 'Delivered' status
        count = sum(1 for row in reader if row['Delivery Status'] == 'Delivered')

    return count

if __name__ == '__main__':
    delivered_count = count_delivered_status(file_path)
    print(f"Number of 'Delivered' statuses {file_path}: {delivered_count}")

