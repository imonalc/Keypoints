# test
python3 extract_keypoints.py --datas Room --descriptors orb --match MNN --points 1000
#python3 extract_keypoints.py --datas Room --descriptors spoint --match BF_KNN
#python3 extract_keypoints.py --datas Room --descriptors sift_P sift_p --match BF_KNN

#python3 extract_keypoints.py --datas Room --descriptors sift spoint
#python3 extract_keypoints.py --datas Room --descriptors spoint_P
#python3 extract_keypoints.py --datas Classroom Realistic Interior1 Interior2 --descriptors orb_p
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_p

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb  --match MNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb --match MNN  --points 1000

## main
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint   --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint   --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P sift_P spoint_P   --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P sift_P spoint_P   --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift tspoint   --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift tspoint   --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_p sift_p spoint_p --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_p sift_p spoint_p --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_a sift_a spoint_a --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_a sift_a spoint_a --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors sphorb --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors sphorb --match BF_KNN  --points 1000

python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint   --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint   --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P sift_P spoint_P   --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P sift_P spoint_P   --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift tspoint   --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift tspoint   --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_p sift_p spoint_p --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_p sift_p spoint_p --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_a sift_a spoint_a --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_a sift_a spoint_a --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors sphorb --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors sphorb --match MNN  --points 1000

echo "完了"