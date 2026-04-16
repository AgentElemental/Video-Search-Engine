import cv2
import os


def extract_frames(video_path: str, output_folder: str, frame_rate: int = 1) -> list[str]:
    """
    Extract frames from a video at the given frame_rate (frames per second).
    Returns a list of saved frame file paths.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Invalid FPS for video: {video_path}")
        cap.release()
        return []

    # How many raw frames to skip between captures
    interval = max(1, int(fps * frame_rate))

    saved_paths = []
    frame_id = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)
            count += 1

        frame_id += 1

    cap.release()
    print(f"Extracted {count} frames from {video_path}")
    return saved_paths
