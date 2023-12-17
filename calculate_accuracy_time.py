import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import math

COMPARE_DESCRIPTORS = ["Proposed", "orb", "sphorb", "sift", "spoint", "alike"]
ABLATION_DESCRIPTORS = ["Proposed","Proposed01", "Proposed1", "Proposed10", "Ltspoint",  "Proposed_un",  "tspoint", "Ftspoint"] # "Proposed20", 
NOT_T_DESCRIPTORS = ["spoint", "tspoint", "sphorb", "Ltspoint", "Proposed01", "Proposed03", "Proposed05", "Proposed1", "Proposed3", "Proposed", "Proposed10", "Proposed20", "Proposed_un", "Ftspoint"] 
METHODS = ["", "t"]
PARAMS = ["R", "T"]
PARAMS_DICT = {"R": "Rotation", "T": "Translation"}
ALL_LOCS = ["Classroom", "Room", "Realistic", "Interior1", "Interior2", "Urban1", "Urban2", "Urban3", "Urban4"]
INDOORS = ["Classroom", "Room", "Realistic", "Interior1", "Interior2"]
OUTDOORS = ["Urban1", "Urban2", "Urban3", "Urban4"]
TIME_CATEGORYS = ["FP", "MC", "PE"]
DESCRIPTORS_DICT = {"Proposed1": "Proposed (1%)",
                    "Proposed01": "Proposed (0.1%)",
                    "Proposed03": "Proposed (0.3%)",
                    "Proposed05": "Proposed (0.5%)",
                    "Proposed3": "Proposed (3%)",
                    "Proposed": "Proposed (5%)",
                    "Proposed10": "Proposed (10%)",
                    "Proposed20": "Proposed (20%)",
                    "Proposed_un": "MNN",
                    "orb": "ORB", 
                    "sphorb": "SPHORB",
                    "sift": "SIFT",
                    "alike": "ALIKE",
                    "spoint": "SPoint",
                    "tspoint": "KNN+L",
                    "Ltspoint": "Proposed (5%)+L",
                    "Ftspoint": "FLANN"
                    }
METHODS_DICT = {"t": "T", "":""}


def read_csv_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [float(row[0]) for row in reader]
    return data

def main():
    for num_points in [500, 1000, 10000]:
        base_path = f"results/FP_{num_points}/values"
        for tgt_idx, tgt in enumerate(["compare", "ablation"]):
            if tgt == "compare":
                descriptors = COMPARE_DESCRIPTORS
            else:
                descriptors = ABLATION_DESCRIPTORS
            fig, axes = plt.subplots(2, 2, figsize=(16, 12),)
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
                            if descriptor in NOT_T_DESCRIPTORS and method == "t":
                                continue
                            all_error_data = []
                            for scene in locs:
                                file_path = f"{base_path}/{scene}_{method}{descriptor}_5PA_GSM_wRT/{param}_ERRORS.csv"
                                error_data = read_csv_data(file_path)
                                all_error_data.extend(error_data)
                            method_tmp = f"{METHODS_DICT[method]}{DESCRIPTORS_DICT[descriptor]}"
                            #print(method_tmp)
                            #print(f"MAE: {np.mean(all_error_data):.3f}, MSE: {np.mean(np.array(all_error_data) ** 2):.3f}")
                            ratios = []
                            for threshold in thresholds:
                                count = np.sum(all_error_data <= threshold * math.pi / 180)
                                ratio = count / len(all_error_data)
                                ratios.append(ratio)
                            ax.plot(thresholds, ratios, linestyle='-', label=method_tmp)


                            if i == 0 and j == 1:
                                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

            plt.tight_layout()
            #plt.show()
            plt.savefig(f"FP{num_points}_{tgt_idx+1}.pdf")
            plt.close()
    

if __name__ == '__main__':
    main()
