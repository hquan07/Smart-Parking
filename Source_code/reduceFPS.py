import cv2
import os

input_path = r"/home/hquan07/detect_car/code prepare parking/7254_overlay.mp4"
output_path = r"/home/hquan07/detect_car/output/7254_overlay_2fps.mp4"

target_fps = 2  # 2 frames per second

# Processing
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {input_path}")

orig_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Original video: {orig_fps:.2f} FPS, {total_frames} frames")

# Writer: write with target_fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

# Calculate the frame step needed
step = max(1, int(round(orig_fps / target_fps)))

frame_idx = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % step == 0:
        out.write(frame)
        saved += 1
    frame_idx += 1

cap.release()
out.release()

print(f"✅ Export complete: {output_path}")
print(f"→ Reduced from {orig_fps:.1f} FPS down to {target_fps} FPS ({saved} frames).")
