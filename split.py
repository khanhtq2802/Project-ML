import csv
import random
from collections import defaultdict

def split_csv(input_file, train_file, test_file, val_file, train_percent, test_percent, val_percent):
    # Read the input CSV file
    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Assuming the first row is header
        data = list(csv_reader)

    # Group data by price range
    price_ranges = {
        'less than 20': [],
        '20 to 30': [],
        '30 to 45': [],
        '45 to 60': [],
        'over 60': []
    }

    for row in data:
        price = float(row[-1])  # Assuming price is the last column
        if price < 20:
            price_ranges['less than 20'].append(row)
        elif 20 <= price < 30:
            price_ranges['20 to 30'].append(row)
        elif 30 <= price < 45:
            price_ranges['30 to 45'].append(row)
        elif 45 <= price < 60:
            price_ranges['45 to 60'].append(row)
        else:
            price_ranges['over 60'].append(row)

    # Split data within each price range
    train_data = []
    test_data = []
    val_data = []

    for price_range, rows in price_ranges.items():
        total_rows = len(rows)
        train_rows = int(total_rows * train_percent)
        test_rows = int(total_rows * test_percent)
        val_rows = total_rows - train_rows - test_rows

        random.shuffle(rows)  # Shuffle the rows within each price range

        train_data.extend(rows[:train_rows])
        test_data.extend(rows[train_rows:train_rows + test_rows])
        val_data.extend(rows[train_rows + test_rows:])

    # Write to train file
    with open(train_file, 'w', newline='') as train_csv:
        train_writer = csv.writer(train_csv)
        train_writer.writerow(header)
        train_writer.writerows(train_data)

    # Write to test file
    with open(test_file, 'w', newline='') as test_csv:
        test_writer = csv.writer(test_csv)
        test_writer.writerow(header)
        test_writer.writerows(test_data)

    # Write to validation file
    with open(val_file, 'w', newline='') as val_csv:
        val_writer = csv.writer(val_csv)
        val_writer.writerow(header)
        val_writer.writerows(val_data)

# Input and output file paths
input_file = 'C:/Users/Minh MPC/Downloads/data_full_cleaned_1.csv'
train_file = 'C:/Users/Minh MPC/Downloads/train_file.csv'
test_file = 'C:/Users/Minh MPC/Downloads/test_file.csv'
val_file = 'C:/Users/Minh MPC/Downloads/val_file.csv'

# Define the percentages of data for training, testing, and validation
train_percent = 0.6  # 60% for training
test_percent = 0.2   # 20% for testing
val_percent = 0.2    # 20% for validation

# Split the CSV file
split_csv(input_file, train_file, test_file, val_file, train_percent, test_percent, val_percent)
