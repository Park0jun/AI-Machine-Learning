# scripts/download_and_extract.py

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pytubefix import YouTube
import mediapipe as mp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_TEST_DIR = os.path.join(ROOT, "data", "test_keypoints")

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_KEYPOINT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def sanitize_filename(name):
    """Windows에서 허용되지 않는 문자 제거"""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

def download_youtube(url):
    yt = YouTube(url)
    name = yt.title.replace(" ", "_")
    name = sanitize_filename(name)  # 특수문자 제거
    filepath = f"{RAW_VIDEO_DIR}/{name}.mp4"
    yt.streams.filter(file_extension='mp4').first().download(
        output_path=RAW_VIDEO_DIR,
        filename=f"{name}.mp4"
    )
    print(f"🎬 Downloaded → {filepath}")
    return filepath, name


def extract_3d_keypoints(video_path, dir_path =RAW_KEYPOINT_DIR, name ="Data"):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_rows = []

    pbar = tqdm(total=total_frames, desc="Extracting BlazePose 3D",
                ascii=True,          # unicode 막대 → ASCII 막대로 변경
                dynamic_ncols=False  # 윈도우 콘솔 버그 방지
                )
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx,
                    "landmark": i,
                    "x": lm.x,
                    "y": -lm.y,      # flip for unity-like coords
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    df = pd.DataFrame(pose_rows)

    # extract local positions from pelvis


    out_path = f"{dir_path}/{name}.npz"
    np.savez(out_path, data=df.to_numpy())
    print(f"📌 Saved 3D keypoints → {out_path}")

def main():
    """
    여러 YouTube 영상에서 키포인트 추출
    URL_LIST에 학습하고 싶은 영상 URL을 추가하세요.
    """
    
    # ============================================================
    # 여기에 학습할 영상 URL들을 추가하세요
    # ============================================================
    URL_LIST = [
        'https://youtube.com/shorts/ENr4S1ZTtv0?si=pb3tqJVtPkMmorqD',
        #'https://youtube.com/shorts/rk36EpJTRDU?si=62q6jZ93FORHOQxa',
        #'https://youtube.com/shorts/EfnHg9EgLpY?si=Cw0xgAs2Bwi3pHN0',
        #'https://youtube.com/shorts/CjG2U5ZH3NE?si=oCyvNgBqjVxuABy9',
        #'https://www.youtube.com/shorts/WBXPN_f7j9g'
        # 더 많은 URL 추가...
    ]
    # ============================================================
    
    print(f"총 {len(URL_LIST)}개 영상 처리 예정\n")
    
    success_count = 0
    fail_list = []
    
    for i, url in enumerate(URL_LIST, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(URL_LIST)}] Processing: {url}")
        print('='*60)
        
        try:
            video_path, name = download_youtube(url)
            extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, name)
            success_count += 1
            print(f"✅ 완료: {name}")
        except Exception as e:
            print(f"❌ 실패: {url}")
            print(f"   에러: {e}")
            fail_list.append(url)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("처리 완료!")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ❌ 실패: {len(fail_list)}개")
    if fail_list:
        print("  실패 목록:")
        for url in fail_list:
            print(f"    - {url}")
    print('='*60)


if __name__ == "__main__":
    main()
