import csv
import os

THRESHOLDS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
DESCRIPTORS = ["orb", "sift", "spoint", "alike", "sphorb"]
METHODS = ["", "t", "c", "cp"]
PARAMS = ["R", "T"]

def calculate_ratio(arr):
    for i, threshold in enumerate(THRESHOLDS):
        cnt = 0
        for j in range(len(arr)):
            if arr[j] < threshold:
                cnt += 1
        print(f"{threshold}:{cnt}, {len(arr)}, {cnt / len(arr)}")


def read_csv_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [float(row[0]) for row in reader]
    return data

def main():     
    base_path = "data/farm/output/"
    for descriptor in DESCRIPTORS:
        for method in METHODS:
            for param in PARAMS:
                all_error_data = []
                for idx in range(1, 6):
                    file_path = f"{base_path}/{method}{descriptor}/pose{idx}/{param}_errors.csv"
                    error_data = read_csv_data(file_path)
                    all_error_data.extend(error_data)
                print(method, descriptor, param)
                calculate_ratio(all_error_data)



if __name__ == '__main__':
    main()