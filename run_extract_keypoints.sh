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

### test ###

python3 extract_keypoints.py --datas atrium_day_10 --descriptors orb tspoint --match MNN --points 1000 --data_folder data_real
#python3 extract_keypoints.py --datas atrium_night_5 hall_night_5 concourse_night_5 piatrium_night_5 --descriptors orb spoint --match MNN --points 1000 --data_folder data_real


### main ###
process_data() {
  echo "Processing INDOORS with $1"
  python3 extract_keypoints.py --datas $INDOORS --descriptors $1 --match $match --points $points
  echo "Processing OUTDOORS with $1"
  python3 extract_keypoints.py --datas $OUTDOORS --descriptors $1 --match $match --points $points
}

#process_data "$descriptors_orb"
#process_data "$descriptors_special"
#process_data "$descriptors_orb"
#process_data "$descriptors_sift"
#process_data "$descriptors_akaze"
#process_data "$descriptors_spoint"
#process_data "$descriptors_aliked"


process_data2() {
  local suffix="$1"
  local descriptors="$2"
  echo "Processing Indoors with interval $suffix and $descriptors"
  python3 extract_keypoints.py --datas "atrium_day_$suffix" "atrium_night_$suffix" "concourse_day_$suffix" "concourse_night_$suffix" --descriptors $descriptors --match $match --points $points --data_folder data_real
  echo "Processing Outdoors with interval $suffix and $descriptors"
  python3 extract_keypoints.py --datas "hall_day_$suffix" "hall_night_$suffix" "piatrium_day_$suffix" "piatrium_night_$suffix" --descriptors $descriptors --match $match --points $points --data_folder data_real
}

FRAME_INTERVAL=2
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_special"
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_sift"
process_data2 "$FRAME_INTERVAL" "$descriptors_akaze"
process_data2 "$FRAME_INTERVAL" "$descriptors_spoint"
process_data2 "$FRAME_INTERVAL" "$descriptors_aliked"

FRAME_INTERVAL=3
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_special"
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_sift"
process_data2 "$FRAME_INTERVAL" "$descriptors_akaze"
process_data2 "$FRAME_INTERVAL" "$descriptors_spoint"
process_data2 "$FRAME_INTERVAL" "$descriptors_aliked"

FRAME_INTERVAL=5
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_special"
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_sift"
process_data2 "$FRAME_INTERVAL" "$descriptors_akaze"
process_data2 "$FRAME_INTERVAL" "$descriptors_spoint"
process_data2 "$FRAME_INTERVAL" "$descriptors_aliked"

FRAME_INTERVAL=10
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_special"
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_sift"
process_data2 "$FRAME_INTERVAL" "$descriptors_akaze"
process_data2 "$FRAME_INTERVAL" "$descriptors_spoint"
process_data2 "$FRAME_INTERVAL" "$descriptors_aliked"

FRAME_INTERVAL=20
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_special"
process_data2 "$FRAME_INTERVAL" "$descriptors_orb"
process_data2 "$FRAME_INTERVAL" "$descriptors_sift"
process_data2 "$FRAME_INTERVAL" "$descriptors_akaze"
process_data2 "$FRAME_INTERVAL" "$descriptors_spoint"
process_data2 "$FRAME_INTERVAL" "$descriptors_aliked"


echo "完了"