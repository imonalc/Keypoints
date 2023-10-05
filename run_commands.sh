#!/bin/bash

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver None

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver SK

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver GSM

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver GSM_wRT

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver GSM_SK

#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver None --inliers "5PA"
#
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver SK --inliers "5PA"
#
#python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver GSM --inliers "5PA"
#
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint alike talike sphorb --solver GSM_wRT --inliers "5PA"
#
python3 extract_keypoints.py --datas Room Classroom Realistic Interior1 Interior2 --descriptors orb torb sift tsift spoint tspoint alike talike sphorb --solver GSM_SK --inliers "5PA"
#
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver None
#
#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver SK

#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver GSM

#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver GSM_wRT

#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver GSM_SK

#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver None --inliers "5PA"

#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver SK --inliers "5PA"

#python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb corb cporb sift tsift csift cpsift spoint tspoint cspoint cpspoint alike talike calike cpalike sphorb --solver GSM --inliers "5PA"

python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint alike talike sphorb --solver GSM_wRT --inliers "5PA"

python3 extract_keypoints.py --datas Urban1 Urban2 Urban3 Urban4 --descriptors orb torb sift tsift spoint tspoint alike talike sphorb --solver GSM_SK --inliers "5PA"


echo "完了"