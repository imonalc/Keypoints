#!/bin/bash

#python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ntspoint --pose pose1 --solver GSM_SK --inliers 5PA
#python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ntspoint --pose pose2 --solver GSM_SK --inliers 5PA
#python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ntspoint --pose pose3 --solver GSM_SK --inliers 5PA
#python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ntspoint --pose pose4 --solver GSM_SK --inliers 5PA
#python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ntspoint --pose pose5 --solver GSM_SK --inliers 5PA

python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Proposed --pose pose1 --solver GSM_SK --inliers 5PA
python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Proposed --pose pose2 --solver GSM_SK --inliers 5PA
python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Proposed --pose pose3 --solver GSM_SK --inliers 5PA
python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Proposed --pose pose4 --solver GSM_SK --inliers 5PA
python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Proposed --pose pose5 --solver GSM_SK --inliers 5PA

echo "完了"