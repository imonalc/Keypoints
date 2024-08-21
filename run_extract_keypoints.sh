# test
#python3 extract_keypoints.py --datas Room --descriptors corb --match MNN --points 1000

INDOORS="Room Classroom Realistic Interior1 Interior2"
OUTDOORS="Urban1 Urban2 Urban3 Urban4"

descriptors_orb="orb corb torb porb rorb"
descriptors_sift="sift csift tsift psift rsift"
descriptors_akaze="akaze cakaze takaze pakaze rakaze"
descriptors_spoint="spoint cspoint tspoint pspoint rspoint"
descriptors_aliked="aliked caliked taliked paliked raliked"
descriptors_special="sphorb loftr spglue"

match="MNN"
points=1000

process_data() {
  echo "Processing $1 with $2"
  python3 extract_keypoints.py --datas $1 --descriptors $2 --match $match --points $points
}

process_data "$INDOORS" "$descriptors_orb"
process_data "$OUTDOORS" "$descriptors_orb"

process_data "$INDOORS" "$descriptors_special"
process_data "$OUTDOORS" "$descriptors_special"

process_data "$INDOORS" "$descriptors_orb"
process_data "$OUTDOORS" "$descriptors_orb"

#process_data "$INDOORS" "$descriptors_sift"
#process_data "$OUTDOORS" "$descriptors_sift"
#
#process_data "$INDOORS" "$descriptors_akaze"
#process_data "$OUTDOORS" "$descriptors_akaze"
#
#process_data "$INDOORS" "$descriptors_spoint"
#process_data "$OUTDOORS" "$descriptors_spoint"
#
#process_data "$INDOORS" "$descriptors_aliked"
#process_data "$OUTDOORS" "$descriptors_aliked"
#


echo "完了"