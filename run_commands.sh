#!/bin/bash

python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Proposed --solver GSM_wRT --inliers "5PA" --points 10000

python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Proposed --solver GSM_wRT --inliers "5PA" --points 10000



echo "完了"