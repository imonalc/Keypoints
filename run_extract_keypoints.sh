# test
python3 extract_keypoints.py --datas Room --descriptors orb rorb torb porb corb akaze --match MNN --points 1000

INDOORS="Room Classroom Realistic Interior1 Interior2"
OUTDOORS="Urban1 Urban2 Urban3 Urban4"

descriptors_set="orb sift spoint akaze aliked"
descriptors_setc="corb csift cakaze cspoint caliked"
descriptors_sett="torb tsift takaze tspoint taliked"
descriptors_setp="porb psift pspoint pakaze paliked"
descriptors_setr="rorb rsift rspoint rakaze raliked"
descriptors_special1="sphorb"

match="MNN"
points=1000

process_data() {
  echo "Processing $1 with $2"
  python3 extract_keypoints.py --datas $1 --descriptors $2 --match $match --points $points
}


process_data "$INDOORS" "$descriptors_set"
process_data "$OUTDOORS" "$descriptors_set"

process_data "$INDOORS" "$descriptors_setc"
process_data "$OUTDOORS" "$descriptors_setc"

process_data "$INDOORS" "$descriptors_sett"
process_data "$OUTDOORS" "$descriptors_sett"

process_data "$INDOORS" "$descriptors_setp"
process_data "$OUTDOORS" "$descriptors_setp"

process_data "$INDOORS" "$descriptors_setr"
process_data "$OUTDOORS" "$descriptors_setr"

process_data "$INDOORS" "$descriptors_special1"
process_data "$OUTDOORS" "$descriptors_special1"

echo "完了"