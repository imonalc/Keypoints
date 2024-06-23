# test
python3 extract_keypoints.py --datas Room --descriptors spoint pspoint tspoint cspoint --match BF_KNN --points 1000
#python3 extract_keypoints.py --datas Room --descriptors spoint --match BF_KNN
#python3 extract_keypoints.py --datas Room --descriptors sift_P sift_p --match BF_KNN

#python3 extract_keypoints.py --datas Room --descriptors sift spoint
#python3 extract_keypoints.py --datas Room --descriptors spoint_P
#python3 extract_keypoints.py --datas Classroom Realistic Interior1 Interior2 --descriptors orb_p
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_p


#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors akaze takaze akaze_P akaze_p --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors akaze takaze akaze_P akaze_p --match BF_KNN  --points 1000

# main
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint akaze aliked  --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint akaze aliked  --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_P sift_P spoint_P akaze_P aliked_P  --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_P sift_P spoint_P akaze_P aliked_P  --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift takaze taliked  --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift takaze taliked --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_p sift_p spoint_p akaze_p aliked_p --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_p sift_p spoint_p akaze_p aliked_p --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors sphorb --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors sphorb --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors tspoint --match BF_KNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors tspoint --match BF_KNN  --points 1000


#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint akaze aliked  --match MNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint akaze aliked  --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors corb csift cakaze cspoint caliked  --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors corb csift cakaze cspoint caliked --match MNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift takaze taliked  --match MNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift takaze taliked --match MNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors porb psift pspoint pakaze paliked --match MNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors porb psift pspoint pakaze paliked --match MNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors sphorb --match MNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors sphorb --match MNN  --points 1000
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors tspoint --match MNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors tspoint --match MNN  --points 1000

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_a sift_a spoint_a --match MNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_a sift_a spoint_a --match MNN  --points 1000

echo "完了"