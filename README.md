# Đề tài niên luận ngành ATTT: PHÁT HIỆN KEYLOGGER SỬ DỤNG THUẬT TOÁN HỌC MÁY
Cấu trúc thư mục đầy đủ của đề tài:
```texttext
D:\Final_keylogger_ML_2\
¦   analyze_data.py
¦   notepad_features_detailed.csv
¦   notepad_features_summary.json
¦   Xu_ly_dl.py
¦   
+---Application
¦       App.py
¦       
+---CatBoost
¦       best_catboost_model.pkl
¦       CB.py
¦       confusion_matrix_best.png
¦       feature_importance.csv
¦       feature_importance.png
¦       training_history.csv
¦       training_history.png
¦       training_log.txt
¦       
+---Dataset_ember_2018
¦   +---ember2018
¦       ¦   ember_dataset_2018_2.tar.bz2
¦       ¦   ember_model_2018.txt
¦       ¦   features.py
¦       ¦   test_features.jsonl
¦       ¦   train_features_0.jsonl
¦       ¦   train_features_1.jsonl
¦       ¦   train_features_2.jsonl
¦       ¦   train_features_3.jsonl
¦       ¦   train_features_4.jsonl
¦       ¦   train_features_5.jsonl
¦       ¦   __init__.py
¦       ¦   
¦       +---__pycache__
¦               features.cpython-310.pyc
¦               __init__.cpython-310.pyc
¦               
+---LightGBM
¦       best_lightgbm_model.pkl
¦       confusion_matrix_best.png
¦       feature_importance.csv
¦       feature_importance.png
¦       LightGBM.py
¦       training_history.csv
¦       training_history.png
¦       training_log.txt
¦       
+---MLP
¦       best_mlp_model.h5
¦       best_mlp_model.keras
¦       confusion_matrix_best.png
¦       MLP.py
¦       scaler.pkl
¦       training_history.csv
¦       training_history.png
¦       training_log.txt
¦       
+---processed_features
¦       byteentropy_features.csv
¦       datadirectories_features.csv
¦       exports_features.csv
¦       general_features.csv
¦       header_features.csv
¦       histogram_features.csv
¦       imports_features.csv
¦       metadata.csv
¦       process_and_split_features.py
¦       section_features.csv
¦       strings_features.csv
¦       
+---processed_features_cleaned
¦       byteentropy_features_cleaned.csv
¦       datadirectories_features_cleaned.csv
¦       exports_features_cleaned.csv
¦       general_features_cleaned.csv
¦       header_features_cleaned.csv
¦       histogram_features_cleaned.csv
¦       imports_features_cleaned.csv
¦       metadata_cleaned.csv
¦       section_features_cleaned.csv
¦       strings_features_cleaned.csv
¦       
+---Random_forest
¦       best_random_forest_model.pkl
¦       confusion_matrix_best.png
¦       feature_importance.csv
¦       feature_importance.png
¦       RF.py
¦       training_history.csv
¦       training_history.png
¦       training_log.txt
¦       
+---visualizations
¦       01_class_distribution.png
¦       02_feature_dimensions.png
¦       03_sparsity_analysis.png
¦       04_variance_analysis.png
¦       05_sample_distributions.png
¦       06_correlation_heatmaps.png
¦       07_pca_analysis.png
¦       08_comprehensive_report.png
¦       analysis_report.txt
¦       analyze_and_visualize_data.py
¦       summary_statistics.csv
¦       
+---XGBoost
¦       best_xgboost_model.pkl
¦       confusion_matrix_best.png
¦       feature_importance.csv
¦       feature_importance.png
¦       training_history.csv
¦       training_history.png
¦       training_log.txt
¦       XGBoost.py
¦       
+---__pycache__
        analyze_and_visualize_data.cpython-310.pyc
```
## 💡 Giới thiệu

Keylogger là loại mã độc nguy hiểm chuyên ghi lại thao tác bàn phím để đánh cắp dữ liệu nhạy cảm. Các phương pháp phát hiện truyền thống dựa trên chữ ký (Signature-based) thường thất bại trước các biến thể mới.

Dự án này đề xuất giải pháp **Học máy (Machine Learning)** kết hợp với **Phân tích tĩnh (Static Analysis)** cấu trúc file PE (Portable Executable) để phát hiện Keylogger mà không cần thực thi chúng, đảm bảo an toàn và hiệu quả cao.

---

## ⚙️ Quy trình thực hiện

Dự án được thực hiện theo quy trình khoa học dữ liệu chặt chẽ gồm 5 bước:

### Bước 1: Thu thập và Xử lý Dữ liệu thô
- **Nguồn dữ liệu:** [EMBER 2018 Dataset](https://github.com/elastic/ember) (1.1 triệu mẫu PE files).
- **Trích xuất:** Sử dụng thư viện `LIEF` để parse cấu trúc file PE.
- **Sàng lọc:** Loại bỏ các mẫu không có nhãn (Unlabeled, nhãn -1), chỉ giữ lại mẫu Lành tính (0) và Độc hại (1).

### Bước 2: Kỹ thuật Đặc trưng (Feature Engineering)
Xử lý làm sạch và tối ưu hóa 2381 đặc trưng đầu vào:
- **Lọc phương sai (Variance Threshold):** Loại bỏ các đặc trưng hằng số (Constant) và phương sai thấp (< 0.005) để giảm nhiễu.
- **Bảo toàn thông tin quan trọng:** Giữ nguyên toàn bộ nhóm đặc trưng **Byte Histogram** và **Byte Entropy** vì tính phân loại cao.
- **Xử lý tương quan:** Loại bỏ các đặc trưng có độ tương quan cao (> 0.95) trong nhóm Header/Section để tránh đa cộng tuyến.
- **Chuẩn hóa:** Áp dụng `StandardScaler` cho mô hình Mạng nơ-ron (MLP).

### Bước 3: Huấn luyện Mô hình (Model Training)
Triển khai huấn luyện 05 thuật toán với các chiến lược tối ưu riêng biệt:
1.  **Random Forest:** Sử dụng chiến lược *Progressive Training* (Tăng dần số cây từ 100 -> 1000).
2.  **XGBoost:** Cấu hình `tree_method='hist'` để tăng tốc trên dữ liệu lớn.
3.  **LightGBM:** Áp dụng chiến lược *Leaf-wise growth*, tối ưu hóa tốc độ và bộ nhớ.
4.  **CatBoost:** Sử dụng `SymmetricTree` và xử lý tốt đặc trưng phân loại.
5.  **MLP (Neural Network):** Kiến trúc mạng hình phễu (1864 -> 1024 -> 512 -> 256 -> 1) với Dropout chống overfitting.

### Bước 4: Đánh giá và So sánh (Evaluation)
- Sử dụng tập kiểm thử độc lập (20% dữ liệu).
- Đánh giá dựa trên 4 chỉ số: **Accuracy, Precision, Recall, F1-Score**.
- Ưu tiên chỉ số **Recall** (Tỷ lệ phát hiện) để giảm thiểu bỏ sót mã độc.

### Bước 5: Xây dựng Ứng dụng Demo (Deployment)
- Xây dựng ứng dụng Desktop bằng **Python Tkinter**.
- **Cơ chế phát hiện:**
    - Tích hợp mô hình tốt nhất (`.pkl`) để quét file.
    - Kết hợp kỹ thuật **Heuristic** (quét từ khóa/DLL nghi vấn).
    - Kết hợp **Behavior Check** (giám sát hành vi IO/CPU bất thường).
- Tích hợp công cụ **Autoruns** để kiểm tra khởi động hệ thống.

---
##🚀 Hướng dẫn Cài đặt & Thực thi
Vui lòng tuân thủ đúng trình tự sau để đảm bảo luồng dữ liệu (Data Pipeline) hoạt động chính xác từ khâu xử lý thô đến huấn luyện mô hình.

###Giai đoạn 1: Xử lý Dữ liệu
Trích xuất đặc trưng: Chạy file analyze_data.py.

Lưu ý: File này hoạt động kết hợp với __init__.py và features.py trong thư mục ember2018. Hãy kiểm tra kỹ đường dẫn thư mục trước khi chạy.

Chuyển đổi định dạng: Chạy process_and_split_features.py để chuyển đổi dữ liệu thô sang các file .csv.

Trực quan hóa (EDA): Chạy analyze_and_visualize_data.py để xem các biểu đồ phân bố dữ liệu sau xử lý.

Làm sạch dữ liệu: Chạy Xu_ly_dl.py. Bước này thực hiện cân bằng dữ liệu, giảm chiều và lọc nhiễu.

###Giai đoạn 2: Huấn luyện Mô hình
Chạy lần lượt các script huấn luyện để tạo ra file model (.pkl hoặc .h5):

python Random_forest/RF.py

python LightGBM/LightGBM.py

python MLP/MLP.py

python XGBoost/XGBoost.py

python CatBoost/CB.py

###Giai đoạn 3: Cấu hình & Chạy Ứng dụng
Cập nhật Model: Mở file App.py, tìm dòng khai báo đường dẫn model và thay thế bằng đường dẫn tới file .pkl tốt nhất vừa huấn luyện (ví dụ: LightGBM/best_lightgbm_model.pkl).

Khởi chạy:

⚠️ Bắt buộc: Chạy App.py dưới quyền Administrator để ứng dụng có thể quét sâu vào Autorun và các tiến trình hệ thống.

🧪 Hướng dẫn Kiểm thử (Testing)
Hệ thống đi kèm file myProject.exe (Keylogger mô phỏng) để phục vụ kiểm thử.

Quy trình Test:
Khởi chạy App.py (Admin).

Chạy file myProject.exe (Admin). Keylogger sẽ bắt đầu ghi nhận phím bấm và lưu log tại thư mục hiện hành.

Trên giao diện App, quan sát cảnh báo hoặc dùng tính năng Quét Mục Khởi Động.

Các phím tắt điều khiển Keylogger (myProject.exe):
Ctrl + Shift + Q: Tắt ứng dụng Keylogger.

Ctrl + K: Kiểm tra trạng thái hoạt động của Keylogger.

##🧹 Hướng dẫn Dọn dẹp sau Kiểm thử
Keylogger mẫu sẽ tạo một khóa Registry để tự khởi động cùng Windows. Sau khi test xong, vui lòng thực hiện các bước sau để xóa bỏ hoàn toàn:

Nhấn tổ hợp phím Win + R.

Nhập lệnh regedit và nhấn OK.

Truy cập đường dẫn sau trên thanh địa chỉ:

Plaintext

Computer\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run
Tìm Value có tên ListenToUser, chuột phải và chọn Delete.
---
