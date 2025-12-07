import sys
import os
import numpy as np
import csv
import json


# Thêm đường dẫn tới thư mục chứa package 'ember2018'
sys.path.insert(0, os.path.join('Dataset_ember_2018'))
from Dataset_ember_2018.ember2018.features import PEFeatureExtractor

# --- Cấu hình ---
file_to_analyze = r"C:\Windows\System32\notepad.exe"
output_csv_path = "notepad_features_detailed.csv"
output_json_path = "notepad_features_summary.json"

print("="*80)
print("EMBER 2018 FEATURE EXTRACTION - VERSION 2 (2381 CHIỀU)")
print("="*80)
print("Khởi tạo trình trích xuất đặc trưng EMBER 2018 Version 2...")
extractor = PEFeatureExtractor(feature_version=2)

print(f"\nBắt đầu phân tích file: {file_to_analyze}")
try:
    with open(file_to_analyze, "rb") as f:
        file_bytes = f.read()

    # Lấy raw features (dictionary format như trong EMBER dataset)
    print("\n⏳ Đang trích xuất raw features...")
    raw_features = extractor.raw_features(file_bytes)
    
    # Lấy vector đặc trưng cuối cùng (2381 chiều cho version 2)
    print("⏳ Đang vector hóa features...")
    feature_vector = extractor.feature_vector(file_bytes)

    print("\n" + "="*80)
    print("✅ PHÂN TÍCH THÀNH CÔNG")
    print("="*80)
    print(f"📦 Kích thước file:        {len(file_bytes):,} bytes ({len(file_bytes)/1024:.2f} KB)")
    print(f"📊 Feature Version:        2 (EMBER 2018)")
    print(f"📏 Tổng số đặc trưng:      {feature_vector.shape[0]} chiều")
    print(f"🔢 Kiểu dữ liệu:           {feature_vector.dtype}")
    
    # --- LƯU RAW FEATURES (JSON) ---
    print(f"\n⏳ Đang lưu raw features vào: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(raw_features, jsonfile, indent=2, default=str)
    print("   ✓ Đã lưu raw features (JSON format)!")
    
    # --- LƯU FEATURE VECTOR CHI TIẾT (CSV) ---
    print(f"\n⏳ Đang lưu feature vector chi tiết vào: {output_csv_path}")
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header cho CSV
        writer.writerow(['FeatureGroup', 'FeatureIndex', 'LocalIndex', 'FeatureName', 'Value'])
        
        current_pos = 0
        
        # Định nghĩa các nhóm đặc trưng theo EMBER 2018 Version 2
        # Tổng: 2381 chiều
        feature_groups = [
            ('ByteHistogram', 256),          # Tần suất byte 0x00-0xFF
            ('ByteEntropyHistogram', 256),   # Entropy histogram
            ('StringExtractor', 104),        # String features (paths, URLs, etc.)
            ('GeneralFileInfo', 10),         # Thông tin file tổng quát
            ('HeaderFileInfo', 62),          # PE Header information
            ('SectionInfo', 255),            # Section table features
            ('ImportsInfo', 1280),           # Import functions (1280 chiều)
            ('ExportsInfo', 128),
            ('Datadirectories', 30)             # Export functions
        ]
        
        print("\n   Đang ghi từng nhóm đặc trưng...")
        for group_name, expected_dim in feature_groups:
            # Lấy slice tương ứng với nhóm này
            feature_slice = feature_vector[current_pos : current_pos + expected_dim]
            
            # Ghi từng giá trị
            for local_idx, value in enumerate(feature_slice):
                global_idx = current_pos + local_idx
                
                # Tạo tên đặc trưng có ý nghĩa
                feature_name = f"{group_name}_{local_idx}"
                
                # Thêm tên cụ thể hơn cho một số nhóm quan trọng
                if group_name == 'ByteHistogram':
                    feature_name = f"Byte_{local_idx:02X}_Count"
                elif group_name == 'ByteEntropyHistogram':
                    feature_name = f"ByteEntropy_Bin_{local_idx}"
                elif group_name == 'GeneralFileInfo':
                    info_names = ['Size', 'VirtualSize', 'HasDebug', 'Exports', 
                                  'Imports', 'HasSignature', 'HasTLS', 'HasResources',
                                  'NumSections', 'Timestamp']
                    if local_idx < len(info_names):
                        feature_name = f"General_{info_names[local_idx]}"
                elif group_name == 'HeaderFileInfo':
                    # Có thể thêm tên cụ thể cho header fields
                    feature_name = f"Header_{local_idx}"
                elif group_name == 'SectionInfo':
                    # Section features
                    feature_name = f"Section_{local_idx}"
                elif group_name == 'ImportsInfo':
                    # Import function features
                    feature_name = f"Import_{local_idx}"
                elif group_name == 'ExportsInfo':
                    # Export function features
                    feature_name = f"Export_{local_idx}"
                
                row = [group_name, global_idx, local_idx, feature_name, value]
                writer.writerow(row)
            
            current_pos += expected_dim
            print(f"      ✓ {group_name:30s}: {expected_dim:4d} features")
        
    print("\n   ✓ Đã lưu feature vector thành công!")
    
    # --- THỐNG KÊ TỔNG QUAN CHI TIẾT ---
    print("\n" + "="*80)
    print("📊 THỐNG KÊ ĐẶC TRƯNG CHI TIẾT")
    print("="*80)
    print(f"{'Nhóm':<30s} {'Số chiều':>10s} {'Non-zero':>10s} {'Sparsity':>10s}")
    print("-"*80)
    
    current_pos = 0
    total_nonzero = 0
    total_dims = 0
    
    for group_name, dim in feature_groups:
        feature_slice = feature_vector[current_pos : current_pos + dim]
        non_zero = np.count_nonzero(feature_slice)
        sparsity = (dim - non_zero) / dim * 100
        
        total_nonzero += non_zero
        total_dims += dim
        
        print(f"{group_name:<30s} {dim:>10d} {non_zero:>10d} {sparsity:>9.1f}%")
        current_pos += dim
    
    print("-"*80)
    print(f"{'TỔNG CỘNG':<30s} {total_dims:>10d} {total_nonzero:>10d} "
          f"{(total_dims-total_nonzero)/total_dims*100:>9.1f}%")
    
    # Thống kê giá trị
    print("\n" + "="*80)
    print("📈 THỐNG KÊ GIÁ TRỊ")
    print("="*80)
    print(f"Min value:     {feature_vector.min():.6f}")
    print(f"Max value:     {feature_vector.max():.6f}")
    print(f"Mean value:    {feature_vector.mean():.6f}")
    print(f"Median value:  {np.median(feature_vector):.6f}")
    print(f"Std value:     {feature_vector.std():.6f}")
    
    # Kiểm tra tổng số chiều
    print("\n" + "="*80)
    print("✅ XÁC MINH FEATURE VERSION")
    print("="*80)
    if current_pos == 2381:
        print(f"✓ ĐÚNG: Feature Version 2 → {current_pos} chiều")
    elif current_pos == 2351:
        print(f"⚠️  Phát hiện: Feature Version 1 → {current_pos} chiều")
    else:
        print(f"⚠️  Cảnh báo: Số chiều không khớp → {current_pos} chiều")
    
    print("\n" + "="*80)
    print("📁 KẾT QUẢ ĐÃ LƯU")
    print("="*80)
    print(f"1. CSV (chi tiết):  {output_csv_path}")
    print(f"   └─ {current_pos:,} dòng (1 header + {current_pos} features)")
    print(f"\n2. JSON (raw):      {output_json_path}")
    print(f"   └─ Dictionary format gốc của EMBER")
    
    # Hiển thị một vài features mẫu
    print("\n" + "="*80)
    print("🔍 MẪU FEATURES (5 đầu tiên)")
    print("="*80)
    for i in range(min(5, len(feature_vector))):
        print(f"Feature {i:4d}: {feature_vector[i]:.6f}")
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH!")
    print("="*80)

except FileNotFoundError:
    print(f"\n❌ LỖI: Không tìm thấy file tại '{file_to_analyze}'")
    print("\n💡 Kiểm tra:")
    print("   1. Đường dẫn file có đúng không?")
    print("   2. File có tồn tại không?")
    print("   3. Có quyền đọc file không?")
    
except ImportError as e:
    print(f"\n❌ LỖI IMPORT: {e}")
    print("\n💡 Kiểm tra:")
    print("   1. Đã cài đặt package ember đúng chưa?")
    print("   2. Thư mục Dataset_ember_2018 có đúng không?")
    print("   3. File features.py có tồn tại trong ember2018/ không?")
    
except Exception as e:
    print(f"\n❌ LỖI: {e}")
    print("\n📋 Chi tiết lỗi:")
    import traceback
    traceback.print_exc()
    
    print("\n💡 Gợi ý:")
    print("   1. Kiểm tra file PE có hợp lệ không")
    print("   2. File có bị corrupt không")
    print("   3. Đủ RAM để xử lý file không")

# Thêm đường dẫn tới thư mục chứa package 'ember2018'
sys.path.insert(0, os.path.join('Dataset_ember_2018'))
from Dataset_ember_2018.ember2018.features import PEFeatureExtractor

# --- Cấu hình ---
file_to_analyze = r"C:\Windows\System32\notepad.exe"
output_csv_path = "notepad_features_detailed.csv"
output_json_path = "notepad_features_summary.json"

print("Khởi tạo trình trích xuất đặc trưng EMBER 2018...")
extractor = PEFeatureExtractor(feature_version=2)

print(f"Bắt đầu phân tích file: {file_to_analyze}")
try:
    with open(file_to_analyze, "rb") as f:
        file_bytes = f.read()

    # Lấy raw features (dictionary format như trong EMBER dataset)
    raw_features = extractor.raw_features(file_bytes)
    
    # Lấy vector đặc trưng cuối cùng (2381 chiều cho version 2)
    feature_vector = extractor.feature_vector(file_bytes)

    print("\n--- PHÂN TÍCH THÀNH CÔNG ---")
    print(f"Kích thước file: {len(file_bytes)} bytes")
    print(f"Tổng số đặc trưng: {feature_vector.shape[0]} chiều")
    
    # --- LƯU RAW FEATURES (JSON) ---
    print(f"\nĐang lưu raw features vào file: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(raw_features, jsonfile, indent=2, default=str)
    print("Đã lưu raw features!")
    
    # --- LƯU FEATURE VECTOR CHI TIẾT (CSV) ---
    print(f"\nĐang lưu feature vector vào file: {output_csv_path}")
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header cho CSV
        writer.writerow(['FeatureGroup', 'FeatureIndex', 'LocalIndex', 'FeatureName', 'Value'])
        
        current_pos = 0
        
        # Định nghĩa tên các nhóm đặc trưng theo thứ tự trong EMBER 2018
        feature_groups = [
            ('ByteHistogram', 256),
            ('ByteEntropyHistogram', 256),
            ('StringExtractor', 104),
            ('GeneralFileInfo', 10),
            ('HeaderFileInfo', 62),
            ('SectionInfo', 255),
            ('ImportsInfo', 1280),
            ('ExportsInfo', 128)
        ]
        
        for group_name, expected_dim in feature_groups:
            # Lấy slice tương ứng
            feature_slice = feature_vector[current_pos : current_pos + expected_dim]
            
            # Ghi từng giá trị
            for local_idx, value in enumerate(feature_slice):
                global_idx = current_pos + local_idx
                
                # Tạo tên đặc trưng có ý nghĩa
                feature_name = f"{group_name}_{local_idx}"
                
                # Thêm tên cụ thể hơn cho một số nhóm
                if group_name == 'ByteHistogram':
                    feature_name = f"Byte_{local_idx:02X}_Count"
                elif group_name == 'ByteEntropyHistogram':
                    feature_name = f"ByteEntropy_Bin_{local_idx}"
                elif group_name == 'GeneralFileInfo':
                    info_names = ['Size', 'VirtualSize', 'HasDebug', 'Exports', 
                                  'Imports', 'HasSignature', 'HasTLS', 'HasResources',
                                  'NumSections', 'Timestamp']
                    if local_idx < len(info_names):
                        feature_name = f"General_{info_names[local_idx]}"
                
                row = [group_name, global_idx, local_idx, feature_name, value]
                writer.writerow(row)
            
            current_pos += expected_dim
        
    print("Đã lưu feature vector thành công!")
    
    # --- THỐNG KÊ TỔNG QUAN ---
    print("\n--- THỐNG KÊ ĐẶC TRƯNG ---")
    current_pos = 0
    for group_name, dim in feature_groups:
        feature_slice = feature_vector[current_pos : current_pos + dim]
        non_zero = np.count_nonzero(feature_slice)
        print(f"{group_name:25s}: {dim:4d} chiều, {non_zero:4d} giá trị khác 0")
        current_pos += dim
    
    print(f"\nTổng cộng: {current_pos} đặc trưng")
    print(f"\nFile CSV: {output_csv_path}")
    print(f"File JSON: {output_json_path}")

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file tại '{file_to_analyze}'.")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
    import traceback
    traceback.print_exc()