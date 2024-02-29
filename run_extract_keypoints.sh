# path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/imonalc/anaconda3/envs/keyglue/lib/

# test
#python3 extract_keypoints.py --datas Room --descriptors orb --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Room --descriptors orb torb sift tsift spoint tspoint sphorb alike talike Ltspoint Ftspoint Proposed5 Proposed01 Proposed1 Proposed10 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 500
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb --solver GSM_wRT --inliers "5PA" --points 1024
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb --solver GSM_wRT --inliers "5PA" --points 1024

# traditional
# indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike --solver GSM_wRT --inliers "5PA" --points 1024
#
## outdoor
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint sphorb alike talike --solver GSM_wRT --inliers "5PA" --points 1024


# Proposed
# indoor
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed2_1 Proposed2_2 Proposed2_3 Proposed2_4 Proposed_nolimit Proposed3_1 Proposed3_2 Proposed3_3 Proposed3_4 Proposed_nolimit2 --solver GSM_wRT --inliers "5PA" --points 1024
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed2_1 Proposed2_2 Proposed2_3 Proposed2_4 Proposed_nolimit Proposed3_1 Proposed3_2 Proposed3_3 Proposed3_4 Proposed_nolimit2 --solver GSM_wRT --inliers "5PA" --points 1024

python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed2_1 Proposed2_2 Proposed2_3 Proposed2_4 Proposed_nolimit Proposed3_1 Proposed3_2 Proposed3_3 Proposed3_4 Proposed_nolimit2 --solver GSM_wRT --inliers "5PA" --points 256
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed2_1 Proposed2_2 Proposed2_3 Proposed2_4 Proposed_nolimit Proposed3_1 Proposed3_2 Proposed3_3 Proposed3_4 Proposed_nolimit2 --solver GSM_wRT --inliers "5PA" --points 256

python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed2_1 Proposed2_2 Proposed2_3 Proposed2_4 Proposed_nolimit Proposed3_1 Proposed3_2 Proposed3_3 Proposed3_4 Proposed_nolimit2 --solver GSM_wRT --inliers "5PA" --points 512
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed2_1 Proposed2_2 Proposed2_3 Proposed2_4 Proposed_nolimit Proposed3_1 Proposed3_2 Proposed3_3 Proposed3_4 Proposed_nolimit2 --solver GSM_wRT --inliers "5PA" --points 512

# outdoor


#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed2_1 Proposed2_2 Proposed2_3 Proposed2_4 Proposed_nolimit --solver GSM_wRT --inliers "5PA" --points 1024


# ablation
# indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed_nolimit Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed_nolimit Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed_nolimit Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 1024
#
## outdoor
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed_nolimit Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 256
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed_nolimit Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 512
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed_nolimit Ltspoint Ftspoint --solver GSM_wRT --inliers "5PA" --points 1024

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