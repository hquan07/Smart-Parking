"""
CÀI ĐẶT TRƯỚC:
# Cài bản PyTorch CPU (vì không có GPU CUDA)
pĐã tìm thấy torch: 2.9.0+cpuip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài Detectron2 tương thích
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Thư viện phụ
pip install opencv-python pycocotools tqdm matplotlib
"""

"""
CẤU TRÚC DỮ LIỆU ĐỀ XUẤT:
dataset_root/
  train/ 
    image1.jpg
    ...
    _annotations.coco.json
  valid/
    image101.jpg
    ...
    _annotations.coco.json
"""

import os
import json
import argparse

from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

# --- HÀM LẤY SỐ LỚP ---
def get_num_classes(coco_json_path):
    """Đọc file JSON và trả về số lượng categories."""
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data.get("categories", []))

# --- HÀM ĐĂNG KÝ DATASET (ĐÃ SỬA LỖI) ---
def register_datasets(train_dir, val_dir=None, train_name="my_train", val_name="my_val"):  # <<< val_dir là tùy chọn
    """
    Đăng ký các bộ dữ liệu train và (tùy chọn) validation.
    """
    # <<< SỬA LỖI LOGIC NỐI CHUỖI:
    train_json = os.path.join(train_dir, "_annotations.coco.json")

    # Kiểm tra sự tồn tại của file annotation
    if not os.path.isfile(train_json):
        raise FileNotFoundError(f"Không tìm thấy file _annotations.coco.json trong thư mục: {train_dir}")

    # Đăng ký dataset train
    register_coco_instances(train_name, {}, train_json, train_dir)

    val_json = None
    # <<< Chỉ đăng ký validation set nếu val_dir được cung cấp
    if val_dir:
        val_json = os.path.join(val_dir, "_annotations.coco.json")  # <<< SỬA LỖI tương tự
        if not os.path.isfile(val_json):
            raise FileNotFoundError(f"Không tìm thấy file _annotations.coco.json trong thư mục: {val_dir}")

        register_coco_instances(val_name, {}, val_json, val_dir)
    else:
        # Nếu không có val_dir, đặt val_name là None
        val_name = None

    return train_json, val_json, train_name, val_name

# --- CLASS TRAINER TÙY CHỈNH ---
class Trainer(DefaultTrainer):
    """
    Trainer tùy chỉnh để sử dụng COCOEvaluator cho quá trình đánh giá.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, tasks=("bbox", "segm"), output_dir=output_folder)

# --- THIẾT LẬP CẤU HÌNH ---
def setup_cfg(args, num_classes, train_name, val_name):  # val_name giờ có thể là None
    """Thiết lập cấu hình Detectron2 từ file YAML và các tham số dòng lệnh."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    # Cấu hình dataset
    cfg.DATASETS.TRAIN = (train_name,)

    # <<< Chỉ thiết lập test set và chu kỳ eval nếu có val_name
    if val_name:
        cfg.DATASETS.TEST = (val_name,)
        cfg.TEST.EVAL_PERIOD = args.eval_period  # Tần suất đánh giá
    else:
        cfg.DATASETS.TEST = ()
        cfg.TEST.EVAL_PERIOD = 0  # Tắt đánh giá trong quá trình training

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Cấu hình Solver
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []

    # Cấu hình CPU training
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.AMP.ENABLED = False

    cfg.OUTPUT_DIR = args.output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SEED = 42

    # Cấu hình Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    return cfg

# --- THAM SỐ DÒNG LỆNH ---
def parse_args():
    """Định nghĩa và phân tích các tham số được truyền vào từ dòng lệnh."""
    p = argparse.ArgumentParser(description="Huấn luyện mô hình Mask R-CNN trên CPU với Detectron2")
    p.add_argument("--train_dir", type=str, required=True, help="Đường dẫn đến thư mục train (ví dụ: data/train)")

    # <<< val_dir không còn bắt buộc (required=False)
    p.add_argument("--val_dir", type=str, required=False, default=None, help="Đường dẫn đến thư mục valid (tùy chọn)")

    p.add_argument("--output", type=str, default="output/maskrcnn_cpu", help="Thư mục để lưu checkpoints và logs")
    p.add_argument("--batch", type=int, default=1, help="Kích thước batch size. Nên để nhỏ (1 hoặc 2) cho CPU.")
    p.add_argument("--base_lr", type=float, default=0.0025, help="Tốc độ học (learning rate)")
    p.add_argument("--max_iter", type=int, default=500, help="Tổng số vòng lặp huấn luyện")
    p.add_argument("--eval_period", type=int, default=100, help="Chu kỳ đánh giá trên tập valid (nếu có)")
    p.add_argument("--resume", action="store_true", help="Cờ để tiếp tục training từ checkpoint cuối cùng nếu có")
    p.add_argument("--num_gpus", type=int, default=0, help="Số GPU sử dụng (đặt là 0 để chỉ dùng CPU)")
    return p.parse_args()

# --- HÀM CHÍNH ---
def main(args):
    """Hàm chính điều phối toàn bộ quy trình."""
    # Đăng ký datasets
    train_json, _, train_name, val_name = register_datasets(args.train_dir, args.val_dir)  # val_dir có thể là None
    num_classes = get_num_classes(train_json)
    print(f"✅ Đã đăng ký dataset thành công. Số lớp (classes): {num_classes}")
    if not val_name:
        print("ℹ️ Không cung cấp thư mục validation, bỏ qua bước đánh giá.")

    # Thiết lập cấu hình
    cfg = setup_cfg(args, num_classes, train_name, val_name)
    default_setup(cfg, args)

    # Khởi tạo Trainer
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    print("\n🚀 Bắt đầu quá trình huấn luyện...")
    trainer.train()
    print("✅ Huấn luyện hoàn tất.")

    # <<< Chỉ chạy đánh giá cuối cùng NẾU có val_name
    if val_name:
        print("\n🧪 Bắt đầu đánh giá trên tập validation...")
        evaluator = COCOEvaluator(val_name, tasks=("bbox", "segm"),
                                  output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
        trainer.test(cfg, trainer.model, evaluators=DatasetEvaluators([evaluator]))
        print("✅ Đánh giá hoàn tất.")
    else:
        print("✅ Hoàn tất! (Bỏ qua đánh giá cuối cùng vì không có validation set)")

# --- ĐIỂM BẮT ĐẦU ---
if __name__ == "__main__":
    args = parse_args()

    print(f"Chạy với {args.num_gpus} GPUs (chế độ CPU).")
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )

"""
python3 train.py \
  --train_dir /home/hquan07/Bouding_box/Dataset/train \
  --batch 1 \
  --max_iter 1000
"""