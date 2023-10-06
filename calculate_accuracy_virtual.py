import csv
import os
import numpy as np
import matplotlib.pyplot as plt

THRESHOLDS = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0]
DESCRIPTORS = ["orb", "sift", "spoint", "sphorb"]
METHODS = ["", "t"]
PARAMS = ["R", "T"]
INDOORS = ["Classroom", "Room", "Realistic", "Interior1", "Interior2"]
OUTDOORS = ["Urban1", "Urban2", "Urban3", "Urban4"]

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
    base_path = "tmp5001000/values"
    for param in PARAMS:
        for loc in ["in", "out"]:
            if loc == "in":
                locs = INDOORS
            else:
                locs = OUTDOORS
            plt.figure(figsize=(10, 6))
            plt.xlabel('Threshold')
            plt.ylabel('Ratio of Values ≤ Threshold')
            plt.title(f'Ratio of Values Below Each Threshold_{loc}_{param}')
            thresholds = np.arange(0, 3.1, 0.1)
            for descriptor in DESCRIPTORS:
                for method in METHODS:
                    if descriptor == "sphorb" and method == "t":
                        continue
                    all_error_data = []
                    for scene in locs:
                        file_path = f"{base_path}/{scene}_{method}{descriptor}_5PA_GSM_wRT/{param}_ERRORS.csv"
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