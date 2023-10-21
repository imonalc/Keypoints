import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import math

DESCRIPTORS = ["Proposed", "orb", "sphorb", "sift", "alike", ]
SPDESCRIPTORS = ["Proposed", "spoint", "Ltspoint",]
METHODS = ["", "t"]
PARAMS = ["R", "T"]
PARAMS_DICT = {"R": "Rotation", "T": "Translation"}
ALL_LOCS = ["Classroom", "Room", "Realistic", "Interior1", "Interior2", "Urban1", "Urban2", "Urban3", "Urban4"]
INDOORS = ["Classroom", "Room", "Realistic", "Interior1", "Interior2"]
OUTDOORS = ["Urban1", "Urban2", "Urban3", "Urban4"]


def read_csv_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [float(row[0]) for row in reader]
    return data

def main():     
    base_path = "result_time2/values"#"tmp5001000/values"
    for tmp in ["FP", "MC", "PE"]:
        print(tmp)
        for descriptor in DESCRIPTORS:
            for method in METHODS:
                if descriptor in ["sphorb", "Ltspoint", "Proposed"] and method == "t":
                    continue
                all_time_data = []
                for scene in ALL_LOCS:
                    file_path = f"{base_path}/{scene}_{method}{descriptor}_5PA_GSM_wRT/TIMES_{tmp}.csv"
                    time_data = read_csv_data(file_path)
                    all_time_data.extend(time_data)
                print(method, descriptor, np.mean(all_time_data))


if __name__ == '__main__':
    main()