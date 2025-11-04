import os
from PIL import Image
import pillow_heif

input_folder = "/home/hquan07/Bouding_box/Ảnh xe sáng.jpg-20251103T115309Z-1-001/Ảnh xe sáng.jpg"
output_folder = "/home/hquan07/Bounding_box/Ảnh xe sáng.jpg-20251103T115309Z-1-001/Ảnh xe sáng"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(input_folder, filename)

        heif_file = pillow_heif.read_heif(heic_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )

        jpg_name = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(output_folder, jpg_name)
        image.save(jpg_path, "JPEG", quality=95)
        print(f"✅ Đã chuyển: {filename} → {jpg_name}")

print("🎉 Hoàn tất chuyển đổi tất cả ảnh HEIC sang JPG.")