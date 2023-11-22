import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import math

COMPARE_DESCRIPTORS = ["Proposed", "orb", "sphorb", "sift", "alike", ]
ABLATION_DESCRIPTORS = ["Proposed", "spoint", "Ltspoint",]
METHODS = ["", "t"]
PARAMS = ["R", "T"]
PARAMS_DICT = {"R": "Rotation", "T": "Translation"}
ALL_LOCS = ["Classroom", "Room", "Realistic", "Interior1", "Interior2", "Urban1", "Urban2", "Urban3", "Urban4"]
INDOORS = ["Classroom", "Room", "Realistic", "Interior1", "Interior2"]
OUTDOORS = ["Urban1", "Urban2", "Urban3", "Urban4"]
TIME_CATEGORYS = ["FP", "MC", "PE"]
DESCRIPTORS_DICT = {"Proposed": "Proposed",
                    "orb": "ORB", 
                    "sphorb": "SPHORB",
                    "sift": "SIFT",
                    "alike": "ALIKE",
                    "spoint": "SPoint",
                    "Ltspoint": "TSPoint+L"}
METHODS_DICT = {"t": "T", "":""}


def read_csv_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [float(row[0]) for row in reader]
    return data

def main():     
    base_path = "results_unlimited/values"
    for tgt in ["compare", "ablation"]:
        if tgt == "compare":
            descriptors = COMPARE_DESCRIPTORS
        else:
            descriptors = ABLATION_DESCRIPTORS
        fig, axes = plt.subplots(2, 2, figsize=(14, 12),)
        plt.rcParams["font.size"] = 20
        for i, param in enumerate(PARAMS):
            for j, loc in enumerate(["indoor", "outdoor"]):
                if loc == "indoor":
                    locs = INDOORS
                else:
                    locs = OUTDOORS
                ax = axes[i, j]
                ax.tick_params(axis='both', labelsize=20)
                ax.set_title(f"{PARAMS_DICT[param]} {loc}")
                ax.grid(True)
                ax.set_ylabel('Ratio of Values ≤ Threshold', fontsize=20)
                ax.set_xlabel('Angle Threshold (°)', fontsize=20)
                thresholds = np.arange(0, 20.1, 0.1)
                for descriptor in descriptors:
                    for method in METHODS:
                        if descriptor in ["sphorb", "Ltspoint", "Proposed"] and method == "t":
                            continue
                        all_error_data = []
                        for scene in locs:
                            file_path = f"{base_path}/{scene}_{method}{descriptor}_5PA_GSM_wRT/{param}_ERRORS.csv"
                            error_data = read_csv_data(file_path)
                            all_error_data.extend(error_data)
                        ratios = []
                        for threshold in thresholds:
                            count = np.sum(all_error_data <= threshold * math.pi / 180)
                            ratio = count / len(all_error_data)
                            ratios.append(ratio)
                        ax.plot(thresholds, ratios, linestyle='-', label=f"{METHODS_DICT[method]}{DESCRIPTORS_DICT[descriptor]}")


                        if i == 0 and j == 1:
                            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        plt.show()

        for time_category in TIME_CATEGORYS:
            print(time_category)
            for descriptor in descriptors:
                for method in METHODS:
                    if descriptor in ["sphorb", "Ltspoint", "Proposed"] and method == "t":
                        continue
                    all_time_data = []
                    for scene in ALL_LOCS:
                        file_path = f"{base_path}/{scene}_{method}{descriptor}_5PA_GSM_wRT/TIMES_{time_category}.csv"
                        time_data = read_csv_data(file_path)
                        all_time_data.extend(time_data)
                    print(method, descriptor, np.mean(all_time_data))
    

if __name__ == '__main__':
    main()