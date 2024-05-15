import csv
import random
import os

def split_csv(input_file, train_file, test_file, train_percent):
    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Assuming the first row is header
        data = list(csv_reader)
          # Shuffle the data randomly

        # Calculate the number of rows for train and test files
        total_rows = len(data)
        train_rows = int(total_rows * train_percent)
        test_rows = total_rows - train_rows

        # Write to train file
        with open(train_file, 'w', newline='') as train_csv:
            train_writer = csv.writer(train_csv)
            train_writer.writerow(header)
            train_writer.writerows(data[:train_rows])

        # Write to test file
        with open(test_file, 'w', newline='') as test_csv:
            test_writer = csv.writer(test_csv)
            test_writer.writerow(header)
            test_writer.writerows(data[train_rows:])

# Input and output file paths
input_file = 'C:/Users/Minh MPC/Downloads/data_full_cleaned_1.csv'
train_file = 'C:/Users/Minh MPC/Downloads/train_file.csv'
test_file = 'C:/Users/Minh MPC/Downloads/test_file.csv'

# Define the percentage of data for training
train_percent = 0.2  # 20% for training, 80% for testing

# Split the CSV file
split_csv(input_file, train_file, test_file, train_percent)
