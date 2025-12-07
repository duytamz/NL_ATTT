# 🛡️ PHÁT HIỆN KEYLOGGER SỬ DỤNG THUẬT TOÁN HỌC MÁY
> **Đề tài Niên luận ngành An toàn Thông tin**

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Sikit%20Learn%20%7C%20TensorFlow-orange)

## 💡 Giới thiệu

Keylogger là loại mã độc nguy hiểm chuyên ghi lại thao tác bàn phím để đánh cắp dữ liệu nhạy cảm. Các phương pháp phát hiện truyền thống dựa trên chữ ký (Signature-based) thường thất bại trước các biến thể mới.

Dự án này đề xuất giải pháp **Học máy (Machine Learning)** kết hợp với **Phân tích tĩnh (Static Analysis)** cấu trúc file PE (Portable Executable) để phát hiện Keylogger mà không cần thực thi chúng, đảm bảo an toàn và hiệu quả cao.

---

## 📂 Cấu trúc thư mục dự án

<details>
<summary><strong>👇 Bấm để xem chi tiết cây thư mục</strong></summary>

```text
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
¦       ¦   ...
¦       ¦   __init__.py
¦       ¦   
¦       +---__pycache__
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
¦       (Các file csv đã làm sạch)
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
```
</details>

---

## ⚙️ Quy trình thực hiện

Dự án được thực hiện theo quy trình khoa học dữ liệu chặt chẽ gồm 5 bước:

### 1️⃣ Thu thập và Xử lý Dữ liệu thô
* **Nguồn dữ liệu:** [EMBER 2018 Dataset](https://github.com/elastic/ember) (1.1 triệu mẫu PE files).
* **Trích xuất:** Sử dụng thư viện `LIEF` để parse cấu trúc file PE.
* **Sàng lọc:** Loại bỏ các mẫu không có nhãn (Unlabeled, nhãn -1), chỉ giữ lại mẫu Lành tính (0) và Độc hại (1).

### 2️⃣ Kỹ thuật Đặc trưng (Feature Engineering)
Xử lý làm sạch và tối ưu hóa **2381 đặc trưng** đầu vào:
* **Lọc phương sai (Variance Threshold):** Loại bỏ các đặc trưng hằng số (Constant) và phương sai thấp (< 0.005) để giảm nhiễu.
* **Bảo toàn thông tin quan trọng:** Giữ nguyên toàn bộ nhóm đặc trưng **Byte Histogram** và **Byte Entropy** vì tính phân loại cao.
* **Xử lý tương quan:** Loại bỏ các đặc trưng có độ tương quan cao (> 0.95) trong nhóm Header/Section để tránh đa cộng tuyến.
* **Chuẩn hóa:** Áp dụng `StandardScaler` cho mô hình Mạng nơ-ron (MLP).

### 3️⃣ Huấn luyện Mô hình (Model Training)
Triển khai huấn luyện 05 thuật toán với các chiến lược tối ưu riêng biệt:

| Mô hình | Chiến lược tối ưu |
| :--- | :--- |
| **Random Forest** | Sử dụng chiến lược *Progressive Training* (Tăng dần số cây từ 100 -> 1000). |
| **XGBoost** | Cấu hình `tree_method='hist'` để tăng tốc trên dữ liệu lớn. |
| **LightGBM** | Áp dụng chiến lược *Leaf-wise growth*, tối ưu hóa tốc độ và bộ nhớ. |
| **CatBoost** | Sử dụng `SymmetricTree` và xử lý tốt đặc trưng phân loại. |
| **MLP (Neural Net)** | Kiến trúc mạng hình phễu (`1864 -> 1024 -> 512 -> 256 -> 1`) với Dropout chống overfitting. |

### 4️⃣ Đánh giá và So sánh (Evaluation)
* Sử dụng tập kiểm thử độc lập (20% dữ liệu).
* Đánh giá dựa trên 4 chỉ số: **Accuracy, Precision, Recall, F1-Score**.
* Ưu tiên chỉ số **Recall** (Tỷ lệ phát hiện) để giảm thiểu bỏ sót mã độc.

### 5️⃣ Xây dựng Ứng dụng Demo (Deployment)
* Xây dựng ứng dụng Desktop bằng **Python Tkinter**.
* **Cơ chế phát hiện:**
    * Tích hợp mô hình tốt nhất (`.pkl`) để quét file.
    * Kết hợp kỹ thuật **Heuristic** (quét từ khóa/DLL nghi vấn).
    * Kết hợp **Behavior Check** (giám sát hành vi IO/CPU bất thường).
* Tích hợp công cụ **Autoruns** để kiểm tra khởi động hệ thống.

---

## 🚀 Hướng dẫn Cài đặt & Thực thi

Vui lòng tuân thủ đúng trình tự sau để đảm bảo luồng dữ liệu (Data Pipeline) hoạt động chính xác.

### Giai đoạn 1: Xử lý Dữ liệu

1.  **Trích xuất đặc trưng:**
    > ⚠️ *Lưu ý: File này hoạt động kết hợp với `__init__.py` và `features.py` trong thư mục `ember2018`.*
    ```bash
    python analyze_data.py
    ```

2.  **Chuyển đổi định dạng:**
    Chuyển đổi dữ liệu thô sang các file `.csv`.
    ```bash
    python processed_features/process_and_split_features.py
    ```

3.  **Trực quan hóa (EDA):**
    Xem các biểu đồ phân bố dữ liệu sau xử lý.
    ```bash
    python visualizations/analyze_and_visualize_data.py
    ```

4.  **Làm sạch dữ liệu:**
    Thực hiện cân bằng dữ liệu, giảm chiều và lọc nhiễu.
    ```bash
    python Xu_ly_dl.py
    ```

### Giai đoạn 2: Huấn luyện Mô hình

Chạy lần lượt các script để tạo ra file model (`.pkl` hoặc `.h5`):

```bash
# Huấn luyện Random Forest
python Random_forest/RF.py

# Huấn luyện LightGBM
python LightGBM/LightGBM.py

# Huấn luyện MLP
python MLP/MLP.py

# Huấn luyện XGBoost
python XGBoost/XGBoost.py

# Huấn luyện CatBoost
python CatBoost/CB.py
```

### Giai đoạn 3: Cấu hình & Chạy Ứng dụng

1.  **Cập nhật Model:** Mở file `App.py`, tìm dòng khai báo đường dẫn model và thay thế bằng đường dẫn tới file `.pkl` tốt nhất vừa huấn luyện (ví dụ: `LightGBM/best_lightgbm_model.pkl`).

2.  **Khởi chạy:**
    > ⚠️ **BẮT BUỘC:** Chạy `App.py` dưới quyền **Administrator** để ứng dụng có thể quét sâu vào Autorun và các tiến trình hệ thống.

---

## 🧪 Hướng dẫn Kiểm thử (Testing)

Hệ thống đi kèm file `myProject.exe` (Keylogger mô phỏng) để phục vụ kiểm thử.

**Quy trình Test:**
1.  Khởi chạy `App.py` (Run as Admin).
2.  Chạy file `myProject.exe` (Run as Admin). Keylogger sẽ bắt đầu ghi nhận phím bấm và lưu log tại thư mục hiện hành.
3.  Trên giao diện App, quan sát cảnh báo hoặc dùng tính năng **Quét Mục Khởi Động**.

**Các phím tắt điều khiển Keylogger (`myProject.exe`):**

| Phím tắt | Chức năng |
| :--- | :--- |
| <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Q</kbd> | Tắt ứng dụng Keylogger |
| <kbd>Ctrl</kbd> + <kbd>K</kbd> | Kiểm tra trạng thái hoạt động của Keylogger |

---

## 🧹 Hướng dẫn Dọn dẹp sau Kiểm thử

Keylogger mẫu sẽ tạo một khóa Registry để tự khởi động cùng Windows. Sau khi test xong, vui lòng thực hiện các bước sau để xóa bỏ hoàn toàn:

1.  Nhấn tổ hợp phím <kbd>Win</kbd> + <kbd>R</kbd>.
2.  Nhập lệnh `regedit` và nhấn **OK**.
3.  Truy cập đường dẫn sau trên thanh địa chỉ:
    ```text
    Computer\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run
    ```
4.  Tìm Value có tên **ListenToUser**, chuột phải và chọn **Delete**.
