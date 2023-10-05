#!/bin/bash

python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb --pose pose1 --solver GSM_SK --inliers 5PA

python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb --pose pose2 --solver GSM_SK --inliers 5PA

python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb --pose pose3 --solver GSM_SK --inliers 5PA

python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb --pose pose4 --solver GSM_SK --inliers 5PA

python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb --pose pose5 --solver GSM_SK --inliers 5PA

echo "完了"