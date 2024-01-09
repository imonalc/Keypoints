#!/bin/bash
#python3 extract_keypoints.py --datas Room --descriptors sphorb --solver GSM_wRT --inliers "5PA" --points 10000
#python3 extract_keypoints_farm.py --descriptors orb --pose pose1 --solver GSM_wRT --inliers 5PA --points 500

## indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed --solver GSM_wRT --inliers "5PA" --points 500
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed --solver GSM_wRT --inliers "5PA" --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed --solver GSM_wRT --inliers "5PA" --points 10000
#
## outdoor
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed --solver GSM_wRT --inliers "5PA" --points 500
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed --solver GSM_wRT --inliers "5PA" --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed --solver GSM_wRT --inliers "5PA" --points 10000


# indoor
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "5PA" --points 500
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "5PA" --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "5PA" --points 10000

# outdoor
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "5PA" --points 500
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "5PA" --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "5PA" --points 10000


#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose1 --solver GSM_wRT --inliers 5PA --points 500
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose2 --solver GSM_wRT --inliers 5PA --points 500
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose3 --solver GSM_wRT --inliers 5PA --points 500
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose4 --solver GSM_wRT --inliers 5PA --points 500
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose5 --solver GSM_wRT --inliers 5PA --points 500
#
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose1 --solver GSM_wRT --inliers 5PA --points 1000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose2 --solver GSM_wRT --inliers 5PA --points 1000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose3 --solver GSM_wRT --inliers 5PA --points 1000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose4 --solver GSM_wRT --inliers 5PA --points 1000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose5 --solver GSM_wRT --inliers 5PA --points 1000
#
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose1 --solver GSM_wRT --inliers 5PA --points 10000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose2 --solver GSM_wRT --inliers 5PA --points 10000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose3 --solver GSM_wRT --inliers 5PA --points 10000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose4 --solver GSM_wRT --inliers 5PA --points 10000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --pose pose5 --solver GSM_wRT --inliers 5PA --points 10000



echo "完了"