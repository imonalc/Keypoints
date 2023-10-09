import csv
import os
import numpy as np
import matplotlib.pyplot as plt

THRESHOLDS = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0]
DESCRIPTORS = ["orb", "sift", "spoint", "sphorb", "alike", "Ntspoint"]
METHODS = ["", "t"]
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
    base_path = "data/Farm/output"
    for param in PARAMS:
        plt.figure(figsize=(10, 6))
        plt.xlabel('Threshold')
        plt.ylabel('Ratio of Values ≤ Threshold')
        plt.title('Ratio of Values Below Each Threshold')
        thresholds = np.arange(0, 8.1, 0.1)
        for descriptor in DESCRIPTORS:
            for method in METHODS:
                if descriptor in ["sphorb", "Ntspoint"] and method == "t":
                    continue
                all_error_data = []
                for idx in range(1, 6):
                    file_path = f"{base_path}/{method}{descriptor}/pose{idx}/{param}_errors.csv"
                    error_data = read_csv_data(file_path)
                    all_error_data.extend(error_data)
                #print(method, descriptor, param)
                #calculate_ratio(all_error_data)
                ratios = []
                for threshold in thresholds:
                    count = np.sum(all_error_data <= threshold)
                    ratio = count / len(all_error_data)
                    ratios.append(ratio)
                plt.plot(thresholds, ratios, marker='.', linestyle='-', label=f"{method}{descriptor}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    main()