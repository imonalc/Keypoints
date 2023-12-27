#!/bin/bash
python3 extract_keypoints.py --datas Room --descriptors sphorb --solver GSM_wRT --inliers "5PA" --points 10000

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
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "8PA" --points 500
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "8PA" --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "8PA" --points 10000

# outdoor
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "8PA" --points 500
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "8PA" --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed Proposed01 Proposed1 Proposed10 Proposed_un --solver GSM_wRT --inliers "8PA" --points 10000

echo "完了"