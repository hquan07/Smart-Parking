# 🧠 Smart Parking – COCO Segmentation with Detectron2

This project is part of a **Smart Parking system** that uses **instance segmentation (Mask R-CNN)** with **Detectron2** to detect and segment vehicles from images or video feeds.

It includes utilities for:
- Training a custom segmentation model using Detectron2  
- Converting `.HEIC` images to `.JPG`  
- Reducing video frame rate for faster processing  
- Detecting and overlaying vehicle segmentation on parking lot videos  

---

## 📁 Project Structure

```
Smart_Parking/
├── Dataset/                         # Custom dataset (COCO format)
│   ├── _annotations.coco.json
│   ├── IMG_4365_jpg.rf.cbdeb8679138d.jpg
│   ├── IMG_4577_jpg.rf.f1f5731b401ae.jpg
│   ├── IMG_4797_jpg.rf.396ee6415ebd12.jpg
│   ├── IMG_5118_jpg.rf.47876f86368adb.
    ├── .....                        # Total: 2,239 images
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
├── output/                          # You can go to drive to download output_train, 
                                     because github has limited the size of uploaded files.
│
├── seg_outputs/                     # Segmentation & summary outputs
│   ├── 7252_detect_overlay.avi
│   ├── 7252_detect_overlay.mp4
│   ├── 7252_seg.csv
│   ├── 7252_summary.txt
│   ├── 7254_detect_overlay.avi
│   ├── 7254_detect_overlay.mp4
│   ├── 7254_seg.csv
│   └── 7254_summary.txt
│
├── Source_code/                     # Main source files
│   ├── Coco_Segmentation.ipynb
│   ├── heic2jpg.py
│   ├── reduceFPS.py
│   └── train.py
│
├── Video_park/                      # Original parking lot videos
│   ├── 7252.mp4
│   └── 7254.mp4
│
├── Video_slots/                     # Slot data and analysis
│   └── 7252_slots.json
│
└── README.md                        # Documentation (this file)
```

---

## ⚙️ Installation

> Works on Ubuntu / macOS / Windows (CPU-only version)

```bash
# 1️⃣ Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # (Linux/macOS)
# or
.venv\Scripts\activate      # (Windows)

# 2️⃣ Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3️⃣ Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 4️⃣ Install additional dependencies
pip install opencv-python pycocotools tqdm matplotlib pillow pillow-heif
```

---

## 🧩 Dataset Format

The dataset must follow the **COCO format**, for example:

```
Dataset/
 ├── IMG_001.jpg
 ├── IMG_002.jpg
 └── _annotations.coco.json
```

Example `_annotations.coco.json` structure:

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [
    {"id": 0, "name": "car"},
    {"id": 1, "name": "motorbike"}
  ]
}
```

---

## 🚀 Training

To train the model, run:

```bash
cd Source_code
python3 train.py   --train_dir ../Dataset   --batch 1   --max_iter 1000
```

### Arguments
| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--train_dir` | `str` | **required** | Path to training dataset |
| `--output` | `str` | `../output/maskrcnn_cpu` | Directory for logs and checkpoints |
| `--batch` | `int` | `1` | Batch size (small for CPU) |
| `--base_lr` | `float` | `0.0025` | Learning rate |
| `--max_iter` | `int` | `500` | Number of training iterations |
| `--eval_period` | `int` | `100` | Evaluation frequency (ignored if no validation set) |
| `--resume` | `flag` | `False` | Resume from the last checkpoint |
| `--num_gpus` | `int` | `0` | Number of GPUs (0 = CPU mode) |

✅ Example output:
```
✅ Datasets registered successfully. Number of classes: 3
🚀 Starting training process...
✅ Training complete.
✅ Finished! (Skipped final evaluation as no validation set was provided)
```

---

## 🖼️ Convert HEIC → JPG

Use `heic2jpg.py` to convert iPhone `.HEIC` images to `.JPG` before training:

```bash
cd Source_code
python3 heic2jpg.py
```

✅ Example output:
```
Converted: IMG_0012.HEIC → IMG_0012.jpg
🎉 All HEIC images successfully converted to JPG.
```

---

## 🎞️ Reduce Video FPS

Use `reduceFPS.py` to reduce the frame rate of parking lot videos to speed up processing.

```bash
cd Source_code
python3 reduceFPS.py
```

A new video with reduced FPS will be saved in the same directory.

---

## 🚗 Detect and Overlay Segmentation

Once the model is trained, use `detect_car+overlay.py` to detect cars and overlay segmentation on videos:

```bash
cd Source_code
python3 detect_car+overlay.py
```

This will:
- Load the trained model from `output/maskrcnn_cpu/`
- Process videos from `Video_park/`
- Save overlay results and statistics into `seg_outputs/`

---

## 🧪 Optional: Notebook Visualization

Open `Coco_Segmentation.ipynb` to:
- Visualize annotations  
- Run model inference interactively  
- Display segmentation masks and bounding boxes  

Run in Jupyter Notebook:
```bash
jupyter notebook Source_code/Coco_Segmentation.ipynb
```

---

## 📞 Contact

**Trần Huy Quân**  
📧 huyquan1607@gmail.com