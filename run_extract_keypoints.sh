# path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/imonalc/anaconda3/envs/keyglue/lib/

# test
#python3 extract_keypoints.py --datas Room --descriptors orb --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Room --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed5 Proposed01 Proposed1 Proposed10 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 500

# image making
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb --solver GSM_wRT --inliers "5PA" --points 1024
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb --solver GSM_wRT --inliers "5PA" --points 1024
#
## traditional
## indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint sphorb alike --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint sphorb alike --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint sphorb alike --solver GSM_wRT --inliers "5PA" --points 1024
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint sphorb alike --solver GSM_wRT --inliers "5PA" --points 10000
#
## outdoor
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint sphorb alike --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint sphorb alike --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint sphorb alike --solver GSM_wRT --inliers "5PA" --points 1024
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint sphorb alike --solver GSM_wRT --inliers "5PA" --points 10000
#
## ablation
## indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 1024
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 10000
#
### outdoor
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 1024
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 10000
#
#
## Proposed
## indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed2_001 Proposed2_003 Proposed2_005 Proposed2_01 Proposed2_02 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 1024
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed2_001 Proposed2_003 Proposed2_005 Proposed2_01 Proposed2_02 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 1024
#
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed2_001 Proposed2_003 Proposed2_005 Proposed2_01 Proposed2_02 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed2_001 Proposed2_003 Proposed2_005 Proposed2_01 Proposed2_02 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 256
#
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed2_001 Proposed2_003 Proposed2_005 Proposed2_01 Proposed2_02 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed2_001 Proposed2_003 Proposed2_005 Proposed2_01 Proposed2_02 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 512
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed2_001 Proposed2_003 Proposed2_005 Proposed2_01 Proposed2_02 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 10000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed2_001 Proposed2_003 Proposed2_005 Proposed2_01 Proposed2_02 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 10000


#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 1024
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 1024
#
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 256
#
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 512
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 512








######
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 Proposed5 --solver GSM_wRT --inliers "5PA" --points 1024
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 Proposed5 --solver GSM_wRT --inliers "5PA" --points 1024
#
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 Proposed5 --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 Proposed5 --solver GSM_wRT --inliers "5PA" --points 256
#
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 Proposed5 --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 Proposed5 --solver GSM_wRT --inliers "5PA" --points 512



# network-base
# indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors sphereglue --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors sphereglue --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors sphereglue --solver GSM_wRT --inliers "5PA" --points 1024
#
## outdoor
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors sphereglue --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors sphereglue --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors sphereglue --solver GSM_wRT --inliers "5PA" --points 1024


## farm
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose1 --solver GSM_wRT --inliers 5PA --points 500
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose2 --solver GSM_wRT --inliers 5PA --points 500
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose3 --solver GSM_wRT --inliers 5PA --points 500
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose4 --solver GSM_wRT --inliers 5PA --points 500
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose5 --solver GSM_wRT --inliers 5PA --points 500
#
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose1 --solver GSM_wRT --inliers 5PA --points 1000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose2 --solver GSM_wRT --inliers 5PA --points 1000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose3 --solver GSM_wRT --inliers 5PA --points 1000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose4 --solver GSM_wRT --inliers 5PA --points 1000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose5 --solver GSM_wRT --inliers 5PA --points 1000
#
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose1 --solver GSM_wRT --inliers 5PA --points 10000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose2 --solver GSM_wRT --inliers 5PA --points 10000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose3 --solver GSM_wRT --inliers 5PA --points 10000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose4 --solver GSM_wRT --inliers 5PA --points 10000
#python3 extract_keypoints_farm.py --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Proposed1 Proposed01 Proposed5 Proposed10 Proposed_nolimit Ltspoint Ftspoint --pose pose5 --solver GSM_wRT --inliers 5PA --points 10000



echo "完了"