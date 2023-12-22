import sys
import os
import os
import cv2

import sys
import pandas as pd
import numpy as np
import argparse

def extract_frames(video_path, output_path, frame_interval, name):
    # 動画ファイルを開く
    video_capture = cv2.VideoCapture(video_path)
    
    # フレームカウンタを初期化
    frame_count = 0
    
    while True:
        # フレームを読み込む
        ret, frame = video_capture.read()
        
        # フレームが正常に読み込まれなかった場合、ループを終了
        if not ret:
            break
        
        # フレームカウンタが指定したフレーム間隔の倍数のときに画像を保存
        if frame_count % frame_interval == 0:
            output_foldername = f"{output_path}/{frame_count//frame_interval}"
            if not os.path.exists(output_foldername):
                os.makedirs(output_foldername)
            output_filename = f"{output_foldername}/{name}.png"
            cv2.imwrite(output_filename, frame)
        
        # フレームカウンタをインクリメント
        if frame_count // frame_interval >=99:
            break
        frame_count += 1
    # メモリを解放し、ファイルを閉じる
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # 使用例
    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--data'      , default="Calibration/pose1")
    args = parser.parse_args()


    video_path = os.path.join(args.data, "O.MP4")
    output_path = os.path.join(args.data)    # 出力画像の保存先ディレクトリパス
    frame_interval = 40            # 画像を切り出すフレーム間隔

    print(video_path)
    print(output_path)

    extract_frames(video_path, output_path, frame_interval, "O")

    video_path = os.path.join(args.data, "R.MP4")
    output_path = os.path.join(args.data)    # 出力画像の保存先ディレクトリパス
    frame_interval = 40       # 画像を切り出すフレーム間隔

    print(video_path)
    print(output_path)

    extract_frames(video_path, output_path, frame_interval, "R")


if __name__ == '__main__':
    main()