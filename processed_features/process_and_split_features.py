import sys
import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# --- CẤU HÌNH ---
# 1. Đảm bảo Python có thể tìm thấy package 'ember2018'
sys.path.insert(0, os.path.join('Dataset_ember_2018'))

try:
    from ember2018.features import PEFeatureExtractor
except ImportError:
    print("LỖI: Không tìm thấy package 'ember2018'.")
    sys.exit(1)

# 2. Đường dẫn theo cấu trúc EMBER 2018 gốc
DATA_DIR = os.path.join("Dataset_ember_2018", "ember2018")
OUTPUT_DIR = "processed_features"

# 3. Kích thước batch xử lý
BATCH_SIZE = 50000

# 4. Danh sách file JSONL gốc của EMBER 2018
# Dataset gốc có các file: train_features_0.jsonl đến train_features_5.jsonl
#                          test_features.jsonl
FILES_TO_PROCESS = [
    "train_features_0.jsonl",
    "train_features_1.jsonl", 
    "train_features_2.jsonl",
    "train_features_3.jsonl",
    "train_features_4.jsonl",
    "train_features_5.jsonl",
    "test_features.jsonl"
]

# Xây dựng đường dẫn đầy đủ
FILES_TO_PROCESS = [os.path.join(DATA_DIR, f) for f in FILES_TO_PROCESS if os.path.exists(os.path.join(DATA_DIR, f))]

# --- CÁC HÀM HỖ TRỢ ---
def count_total_records(file_paths):
    """Đếm tổng số bản ghi trong tất cả các file"""
    total = 0
    print("Đang đếm tổng số bản ghi...")
    for fp in file_paths:
        if os.path.exists(fp):
            with open(fp, 'r') as f:
                total += sum(1 for _ in f)
    return total

def process_and_write_batch(batch_records, writers, metadata_writer, extractor, is_first_batch):
    """
    Vector hóa một batch và ghi vào các file CSV
    """
    metadata_list = []
    vectorized_features_list = []

    # Xử lý từng record trong batch
    for record in batch_records:
        # Lấy metadata theo cấu trúc EMBER 2018
        meta = {
            "sha256": record.get("sha256", ""),
            "label": record.get("label", -1),  # -1 = unlabeled, 0 = benign, 1 = malicious
            "appeared": record.get("appeared", ""),  # train hoặc test
            "avclass": record.get("avclass", "")
        }
        metadata_list.append(meta)
        
        # Vector hóa đặc trưng (2381 chiều cho version 2)
        try:
            vector = extractor.process_raw_features(record)
        except Exception as e:
            # Nếu lỗi, tạo vector zero
            print(f"Cảnh báo: Không thể xử lý record {meta['sha256']}: {e}")
            vector = np.zeros(extractor.dim, dtype=np.float32)
        
        vectorized_features_list.append(vector)

    # Chuyển đổi thành DataFrame và numpy array
    metadata_df = pd.DataFrame(metadata_list)
    feature_matrix = np.array(vectorized_features_list, dtype=np.float32)

    # Ghi metadata
    metadata_df.to_csv(metadata_writer, index=False, header=is_first_batch, mode='a' if not is_first_batch else 'w')

    # Ghi từng nhóm đặc trưng
    current_pos = 0
    for feature_group in extractor.features:
        name = feature_group.name
        dim = feature_group.dim
        
        # Trích xuất đặc trưng của nhóm này
        group_data = feature_matrix[:, current_pos: current_pos + dim]
        
        # Tạo tên cột theo format chuẩn
        columns = [f"{name}_{i}" for i in range(dim)]
        group_df = pd.DataFrame(group_data, columns=columns)
        
        # Ghi vào file (append mode)
        group_df.to_csv(writers[name], index=False, header=is_first_batch, mode='a' if not is_first_batch else 'w')
        
        current_pos += dim

def print_dataset_info(extractor, version):
    """In thông tin về cấu trúc dataset"""
    print("\n" + "="*70)
    print("THÔNG TIN CẤU TRÚC ĐẶC TRƯNG EMBER 2018")
    print("="*70)
    print(f"Feature Version: {version}")
    print(f"Tổng số chiều: {extractor.dim}")
    print("\nChi tiết các nhóm đặc trưng:")
    print("-"*70)
    
    current_pos = 0
    for feature_group in extractor.features:
        print(f"{feature_group.name:30s} | Chiều: {feature_group.dim:4d} | Vị trí: {current_pos:4d}-{current_pos + feature_group.dim - 1:4d}")
        current_pos += feature_group.dim
    print("="*70 + "\n")

# --- CHƯƠNG TRÌNH CHÍNH ---
def main():
    print("="*70)
    print("EMBER 2018 FEATURE EXTRACTION & PROCESSING")
    print("="*70)
    
    # Kiểm tra file tồn tại
    if not FILES_TO_PROCESS:
        print("LỖI: Không tìm thấy file dữ liệu nào trong thư mục:")
        print(f"  {DATA_DIR}")
        print("\nCác file cần có:")
        for f in ["train_features_0.jsonl", "train_features_1.jsonl", "train_features_2.jsonl",
                  "train_features_3.jsonl", "train_features_4.jsonl", "train_features_5.jsonl",
                  "test_features.jsonl"]:
            print(f"  - {f}")
        sys.exit(1)
    
    print(f"\nĐã tìm thấy {len(FILES_TO_PROCESS)} file(s) để xử lý:")
    for fp in FILES_TO_PROCESS:
        print(f"  - {os.path.basename(fp)}")
    
    # Tạo thư mục output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"\nĐã tạo thư mục đầu ra: {OUTPUT_DIR}")

    # Khởi tạo extractor
    feature_version = 2
    print(f"\nĐang khởi tạo PEFeatureExtractor (version {feature_version})...")
    extractor = PEFeatureExtractor(feature_version=feature_version)
    print_dataset_info(extractor, feature_version)
    
    # Đếm tổng số records
    total_records = count_total_records(FILES_TO_PROCESS)
    print(f"Tổng số bản ghi: {total_records:,}")
    
    # --- Mở tất cả file CSV đầu ra ---
    print(f"\nChuẩn bị {len(extractor.features) + 1} file CSV đầu ra...")
    writers = {}
    file_handles = []
    
    # File metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    metadata_writer = open(metadata_path, 'w', newline='', encoding='utf-8')
    file_handles.append(metadata_writer)

    # File cho từng nhóm đặc trưng
    for feature_group in extractor.features:
        name = feature_group.name
        output_path = os.path.join(OUTPUT_DIR, f"{name}_features.csv")
        f = open(output_path, 'w', newline='', encoding='utf-8')
        writers[name] = f
        file_handles.append(f)
    
    print("Đã mở tất cả các file đầu ra.")

    # --- Xử lý dữ liệu theo batch ---
    print(f"\nBắt đầu xử lý với batch size = {BATCH_SIZE:,}")
    print("-"*70)
    
    batch_records = []
    is_first_batch = True
    
    with tqdm(total=total_records, desc="Xử lý", unit="records") as pbar:
        for file_path in FILES_TO_PROCESS:
            with open(file_path, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        batch_records.append(record)
                        
                        # Khi batch đầy, xử lý
                        if len(batch_records) >= BATCH_SIZE:
                            process_and_write_batch(batch_records, writers, metadata_writer, extractor, is_first_batch)
                            batch_records = []
                            is_first_batch = False
                        
                        pbar.update(1)
                    except json.JSONDecodeError as e:
                        print(f"\nCảnh báo: Không thể parse JSON: {e}")
                        continue

    # Xử lý batch cuối cùng
    if batch_records:
        process_and_write_batch(batch_records, writers, metadata_writer, extractor, is_first_batch)
        
    # --- Đóng tất cả file ---
    print("\n\nĐang đóng các file...")
    for f in file_handles:
        f.close()
        
    # --- Thống kê kết quả ---
    print("\n" + "="*70)
    print("KẾT QUẢ XỬ LÝ")
    print("="*70)
    print(f"Thư mục đầu ra: {OUTPUT_DIR}")
    print("\nCác file đã tạo:")
    print(f"  1. metadata.csv")
    for i, feature_group in enumerate(extractor.features, 2):
        print(f"  {i}. {feature_group.name}_features.csv ({feature_group.dim} cột)")
    
    # Kiểm tra kích thước file
    print("\nKích thước file:")
    total_size = 0
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        size = os.path.getsize(filepath) / (1024**2)  # MB
        total_size += size
        print(f"  {filename:40s}: {size:8.2f} MB")
    print(f"  {'TỔNG CỘNG':40s}: {total_size:8.2f} MB")
    
    print("\n" + "="*70)
    print("HOÀN THÀNH!")
    print("="*70)

if __name__ == "__main__":
    main()