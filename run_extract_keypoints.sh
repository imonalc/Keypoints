# path
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/imonalc/anaconda3/envs/keyglue/lib/

# test
#python3 extract_keypoints.py --datas Room --descriptors orb
#python3 extract_keypoints.py --datas Room --descriptors sift spoint
#python3 extract_keypoints.py --datas Room --descriptors spoint_P
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb 
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb  






## main
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint   --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint   --match BF_KNN

python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P sift_P spoint_P   --match BF_KNN
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P sift_P spoint_P   --match BF_KNN

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift tspoint   --match BF_KNN
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift tspoint   --match BF_KNN

python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint   --match MNN
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint   --match MNN
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P sift_P spoint_P   --match MNN
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P sift_P spoint_P   --match MNN
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift tspoint   --match MNN
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift tspoint   --match MNN



echo "完了"