# path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/imonalc/anaconda3/envs/keyglue/lib/

# test
#python3 extract_keypoints.py --datas Room --descriptors orb --solver GSM_wRT  --points 10000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors spoint spoint_P1 spoint_P2 spoint_P3 spoint_P4 --solver GSM_wRT  --match BF_KNN
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors spoint spoint_P1 spoint_P2 spoint_P3 spoint_P4 --solver GSM_wRT --match BF_KNN
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift tspoint --solver GSM_wRT  --match BF_KNN
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift tspoint --solver GSM_wRT  --match BF_KNN

python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P3 sift_P3 --solver GSM_wRT  --match BF_KNN
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P3 sift_P3 --solver GSM_wRT  --match BF_KNN
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift --solver GSM_wRT  --match BF_KNN
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift --solver GSM_wRT  --match BF_KNN

#
## traditional
## indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint alike --solver GSM_wRT  --points 10000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint alike --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift alike --solver GSM_wRT  --match MNN
#
## outdoor

#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint alike --solver GSM_wRT  --points 10000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint alike --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift alike --solver GSM_wRT  --match MNN
#
## ablation
## indoor
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 tspoint --solver GSM_wRT  --points 10000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 tspoint --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 tspoint --solver GSM_wRT  --match MNN
#
### outdoor
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 tspoint --solver GSM_wRT  --points 10000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 tspoint --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors Proposed1 Proposed2 Proposed3 Proposed4 tspoint --solver GSM_wRT  --match MNN




## farm
#python3 extract_keypoints_farm.py --descriptors orb --scene Room --pose pose1 --solver GSM_wRT  --match BF
#python3 extract_keypoints_farm.py --descriptors orb sift alike Proposed1 Proposed2 Proposed3 Proposed4 tspoint  --scene Room --pose pose1 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints_farm.py --descriptors orb sift alike Proposed1 Proposed2 Proposed3 Proposed4 tspoint  --scene Room --pose pose1 --solver GSM_wRT  --match MNN

#python3 extract_keypoints_farm.py --descriptors orb sift alike Proposed1 Proposed2 Proposed3 Proposed4 tspoint  --scene Corridor --pose pose1 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints_farm.py --descriptors orb sift alike Proposed1 Proposed2 Proposed3 Proposed4 tspoint  --scene Corridor --pose pose1 --solver GSM_wRT  --match MNN

#python3 extract_keypoints_farm.py --descriptors orb sift alike Proposed1 Proposed2 Proposed3 Proposed4 tspoint  --scene Urban --pose pose1 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints_farm.py --descriptors orb sift alike Proposed1 Proposed2 Proposed3 Proposed4 tspoint  --scene Urban --pose pose1 --solver GSM_wRT  --match MNN

#python3 extract_keypoints.py --datas Room --descriptors orb_P4 spoint_P4 --solver GSM_wRT  --match BF_KNN

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P4 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P4 --solver GSM_wRT  --match MNN
#
### outdoor
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P4 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P4 --solver GSM_wRT  --match MNN
## farm
#python3 extract_keypoints_farm.py --descriptors orb_P4  --scene Room --pose pose1 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints_farm.py --descriptors orb_P4  --scene Room --pose pose1 --solver GSM_wRT  --match MNN
#python3 extract_keypoints_farm.py --descriptors orb_P4  --scene Corridor --pose pose1 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints_farm.py --descriptors orb_P4  --scene Corridor --pose pose1 --solver GSM_wRT  --match MNN
#python3 extract_keypoints_farm.py --descriptors orb_P4  --scene Urban --pose pose1 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints_farm.py --descriptors orb_P4 --scene Urban --pose pose1 --solver GSM_wRT  --match MNN


echo "完了"