# test
python3 extract_keypoints.py --datas Room --descriptors orb rorb rspoint --match MNN --points 1000


python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb sift spoint akaze aliked  --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb sift spoint akaze aliked  --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors corb csift cakaze cspoint caliked  --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors corb csift cakaze cspoint caliked --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors torb tsift takaze taliked  --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors torb tsift takaze taliked --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors porb psift pspoint pakaze paliked --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors porb psift pspoint pakaze paliked --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors rorb rsift rspoint rakaze raliked --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors rorb rsift rspoint rakaze raliked --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors sphorb --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors sphorb --match MNN  --points 1000
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors tspoint --match MNN  --points 1000
python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors tspoint --match MNN  --points 1000

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb_a sift_a spoint_a --match MNN  --points 1000
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb_a sift_a spoint_a --match MNN  --points 1000

echo "完了"