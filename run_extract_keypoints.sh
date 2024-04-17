# path
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/imonalc/anaconda3/envs/keyglue/lib/

# test
python3 extract_keypoints.py --datas Room --descriptors orb --solver GSM_wRT  --points 10000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors spoint spoint_P1 spoint_P2 spoint_P3 spoint_P4 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb --solver GSM_wRT  --points 10000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb --solver GSM_wRT 

#python3 extract_keypoints.py --datas Room --descriptors sift_P --solver GSM_wRT  --points 10000 --match BF_KNN

#python3 extract_keypoints.py --datas Classroom --descriptors tspoint --solver GSM_wRT  --points 10000
## main
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors spoint spoint_P1 spoint_P2 spoint_P3 spoint_P4 --solver GSM_wRT --match BF_KNN
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint --solver GSM_wRT  --match BF_KNN

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P sift_P spoint_P --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P sift_P spoint_P --solver GSM_wRT  --match BF_KNN

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift tspoint --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift tspoint --solver GSM_wRT  --match BF_KNN
#
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P3 sift_P3 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P3 sift_P3 --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift --solver GSM_wRT  --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift --solver GSM_wRT  --match BF_KNN




echo "完了"