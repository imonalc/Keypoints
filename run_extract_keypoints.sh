INDOORS="Room Classroom Realistic Interior1 Interior2"
OUTDOORS="Urban1 Urban2 Urban3 Urban4"
DAYS="atrium_day hall_day concourse_day piatrium_day"
NIGHTS="atrium_night hall_night concourse_night piatrium_night"

descriptors_orb="orb corb torb porb rorb"
descriptors_sift="sift csift tsift psift rsift"
descriptors_akaze="akaze cakaze takaze pakaze rakaze"
descriptors_spoint="spoint cspoint tspoint pspoint rspoint"
descriptors_aliked="aliked caliked taliked paliked raliked"
descriptors_special="sphorb loftr spglue"

match="MNN"
points=1000

### test ###

python3 extract_keypoints.py --datas atrium_day hall_day concourse_day piatrium_day --descriptors orb spoint --match MNN --points 1000
python3 extract_keypoints.py --datas atrium_night hall_night concourse_night piatrium_night --descriptors orb spoint --match MNN --points 1000


### main ###
process_data() {
  echo "Processing $1 with $2"
  python3 extract_keypoints.py --datas $1 --descriptors $2 --match $match --points $points
}

#process_data "$INDOORS" "$descriptors_orb"
#process_data "$OUTDOORS" "$descriptors_orb"
#
#process_data "$INDOORS" "$descriptors_special"
#process_data "$OUTDOORS" "$descriptors_special"
#
#process_data "$INDOORS" "$descriptors_orb"
#process_data "$OUTDOORS" "$descriptors_orb"

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

process_data "$DAYS" "$descriptors_orb"
process_data "$NIGHTS" "$descriptors_orb"

process_data "$DAYS" "$descriptors_special"
process_data "$NIGHTS" "$descriptors_special"

process_data "$DAYS" "$descriptors_orb"
process_data "$NIGHTS" "$descriptors_orb"

process_data "$DAYS" "$descriptors_sift"
process_data "$NIGHTS" "$descriptors_sift"

process_data "$DAYS" "$descriptors_akaze"
process_data "$NIGHTS" "$descriptors_akaze"

process_data "$DAYS" "$descriptors_spoint"
process_data "$NIGHTS" "$descriptors_spoint"

process_data "$DAYS" "$descriptors_aliked"
process_data "$NIGHTS" "$descriptors_aliked"



echo "完了"