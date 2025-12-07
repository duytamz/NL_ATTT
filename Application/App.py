import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import time
import joblib
import numpy as np
import psutil
import os
import subprocess
import ctypes
import sys
import urllib.request
import zipfile
import io

# ===== CONFIG =====
MODEL_PATH = r"D:/Final_keylogger_ML_2/LightGBM/best_lightgbm_model.pkl"  # model RandomForest (.pkl)
POLL_INTERVAL = 5.0    # giây giữa các lần quét
ALERT_COOLDOWN = 600.0  # 10 phút = 600 giây: thời gian tối thiểu giữa 2 cảnh báo cho cùng 1 exe
LOG_LIMIT = 200        # số dòng log tối đa hiển thị trong GUI

# --- CẢI TIẾN: Cấu hình cho Autoruns ---
AUTORUNS_URL = "https://download.sysinternals.com/files/Autoruns.zip"
AUTORUNS_DIR = os.path.join(os.getenv("APPDATA"), "MalwareDetector", "Autoruns")
AUTORUNS_EXE_PATH = os.path.join(AUTORUNS_DIR, "Autoruns.exe")

# ===== GLOBAL STATE =====
running = False
clf = None
MODEL_EXPECTED_FEATURES = None
last_alerts = {}  # map key -> timestamp của lần cảnh báo cuối cùng; key = exe_path nếu có, else pid

SYSTEM_USER_KEYS = {"nt authority\\system", "system", "local system", "nt authority\\localsystem"}
WINDOWS_DIR = os.environ.get("WINDIR", r"C:\Windows").lower()

# ====== 1️⃣ YÊU CẦU QUYỀN ADMIN ======
def ensure_admin():
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            messagebox.showwarning("Yêu cầu quyền Administrator",
                                   "Vui lòng chạy chương trình bằng quyền Administrator để quét toàn bộ tiến trình và chạy các công cụ nâng cao.")
            sys.exit(1)
    except Exception:
        messagebox.showwarning("Kiểm tra quyền thất bại",
                               "Không thể xác định quyền admin; một số chức năng có thể không hoạt động.")

# ====== 2️⃣ LOAD MODEL ======
def load_model():
    global clf, MODEL_EXPECTED_FEATURES
    try:
        clf = joblib.load(MODEL_PATH)
        MODEL_EXPECTED_FEATURES = getattr(clf, "n_features_in_", None)
        safe_log(f"[+] Loaded model: {MODEL_PATH}, expects {MODEL_EXPECTED_FEATURES} features")
    except Exception as e:
        safe_log(f"[-] Không thể load model: {e}")
        clf = None
        MODEL_EXPECTED_FEATURES = None

# ====== 3️⃣ EXTRACTOR EMBER (placeholder) ======
def extract_ember_feature_vector(exe_path):
    """
    ⚠️ PHẢI THAY BẰNG HÀM THỰC TẾ nếu bạn có script EMBER feature extractor.
    """
    raise NotImplementedError(
        "Chưa có hàm extract_ember_feature_vector(exe_path).\n"
        "Hãy implement để trích xuất 2381 đặc trưng từ file exe_path giống như khi huấn luyện model EMBER."
    )

# ====== 4️⃣ HEURISTIC + MODULE CHECK ======
SUSPICIOUS_KEYWORDS = ["dll32", "hook", "keylog", "keylogger", "logger", "kbd", "spy", "hook32"]

def list_loaded_modules(pid):
    try:
        cmd = f'wmic process where processid={pid} get CommandLine /format:list'
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return out.lower()
    except Exception:
        return ""

def heuristic_check_process(proc):
    try:
        name = (proc.name() or "").lower()
        if any(k in name for k in SUSPICIOUS_KEYWORDS):
            return True
        pid = proc.pid
        modules_info = list_loaded_modules(pid)
        if any(k in modules_info for k in SUSPICIOUS_KEYWORDS):
            return True
    except Exception:
        pass
    return False

# ====== 5️⃣ HELPER: hệ thống vs user process ======
def is_system_process(proc, exe_path=None):
    try:
        if not exe_path:
            exe_path = proc.info.get("exe") if isinstance(proc, psutil.Process) else None
        if exe_path and exe_path.lower().startswith(WINDOWS_DIR):
            return True
        try:
            uname = proc.username() or ""
            if uname.lower() in SYSTEM_USER_KEYS:
                return True
        except Exception:
            pass
        name = (proc.name() or "").lower()
        if name in {"svchost.exe", "lsass.exe", "wininit.exe", "services.exe", "csrss.exe"}:
            return True
    except Exception:
        pass
    return False

# ====== 6️⃣ BEHAVIORAL CHECK ======
def behavior_check(proc):
    try:
        cpu = proc.cpu_percent(interval=0.1)
        io = proc.io_counters()
        if cpu < 1.0 and io.write_bytes > 2_000_000:
            return True
    except Exception:
        pass
    return False

# ====== 7️⃣ THREAD-SAFE GUI HELPERS ======
def safe_log(msg):
    timestamped = f"{time.strftime('%H:%M:%S')} - {msg}"
    print(timestamped)
    try:
        if 'app' in globals() and app.winfo_exists():
            def append():
                try:
                    txt_log.config(state="normal")
                    txt_log.insert(tk.END, timestamped + "\n")
                    txt_log.config(state="disabled")
                    txt_log.yview_moveto(1.0)
                    lines = int(txt_log.index('end-1c').split('.')[0])
                    if lines > LOG_LIMIT:
                        txt_log.config(state="normal")
                        txt_log.delete("1.0", f"{lines - LOG_LIMIT}.0")
                        txt_log.config(state="disabled")
                except Exception:
                    pass
            app.after(0, append)
    except Exception:
        pass

def show_alert(title, msg):
    try:
        if 'app' in globals() and app.winfo_exists():
            app.after(0, lambda: messagebox.showwarning(title, msg))
        else:
            print(f"[ALERT] {title}: {msg}")
    except Exception:
        pass

# ====== 8️⃣ DETECTION LOOP ======
def should_alert_for_key(key):
    now = time.time()
    last = last_alerts.get(key, 0)
    if now - last > ALERT_COOLDOWN:
        last_alerts[key] = now
        return True
    return False

def detect_loop():
    global running
    safe_log("🟢 Bắt đầu vòng quét phát hiện tiến trình...")
    while running:
        try:
            for proc in psutil.process_iter(attrs=["pid", "name", "exe"]):
                try:
                    if not running: break
                    pid = proc.info.get("pid")
                    name = proc.info.get("name") or ""
                    exe = proc.info.get("exe")
                    if not exe or is_system_process(proc, exe):
                        continue
                    key = exe.lower()
                    
                    # Model-based detection
                    if clf is not None:
                        try:
                            vec = extract_ember_feature_vector(exe)
                            X = np.array(vec, dtype=np.float32).reshape(1, -1)
                            if X.shape[1] == MODEL_EXPECTED_FEATURES and int(clf.predict(X)[0]) == 1:
                                if should_alert_for_key(key):
                                    msg = f"🔴 Model phát hiện tiến trình khả nghi: {name} (PID={pid})"
                                    safe_log(msg + f" Path: {exe}")
                                    show_alert("⚠️ Phát hiện nghi ngờ (Model)", msg + f"\nPath: {exe}")
                        except NotImplementedError: pass
                        except Exception as e: safe_log(f"Lỗi khi predict {name}: {e}")
                    
                    # Heuristic detection
                    if heuristic_check_process(proc):
                        if should_alert_for_key(key):
                            msg = f"🟠 Heuristic cảnh báo: {name} (PID={pid}) có DLL/keyword khả nghi."
                            safe_log(msg + f" Path: {exe}")
                            show_alert("⚠️ Cảnh báo Heuristic", msg + f"\nPath: {exe}")

                    # Behavior detection
                    if behavior_check(proc):
                         if should_alert_for_key(key):
                            msg = f"🟡 Behavior cảnh báo: {name} (PID={pid}) ghi file bất thường."
                            safe_log(msg + f" Path: {exe}")
                            show_alert("⚠️ Hành vi nghi ngờ", msg + f"\nPath: {exe}")

                except psutil.NoSuchProcess: continue
                except Exception as e: safe_log(f"Lỗi xử lý tiến trình: {e}")
            if not running: break
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            safe_log(f"Lỗi vòng quét chính: {e}")
            time.sleep(POLL_INTERVAL)
    safe_log("🔴 Vòng quét đã dừng.")

# --- CẢI TIẾN: Chức năng quét mục khởi động với Autoruns ---
def download_and_unzip_autoruns():
    """Tải và giải nén Autoruns vào thư mục AppData."""
    try:
        safe_log("[+] Đang kiểm tra Autoruns...")
        if os.path.exists(AUTORUNS_EXE_PATH):
            safe_log("[+] Autoruns đã có sẵn.")
            return True

        safe_log("[-] Không tìm thấy Autoruns. Bắt đầu tải xuống...")
        os.makedirs(AUTORUNS_DIR, exist_ok=True)
        
        with urllib.request.urlopen(AUTORUNS_URL) as response:
            if response.status != 200:
                safe_log(f"[-] Lỗi tải xuống: HTTP {response.status}")
                return False
            zip_content = response.read()

        safe_log("[+] Tải xuống hoàn tất. Đang giải nén...")
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            zf.extractall(AUTORUNS_DIR)
        
        if os.path.exists(AUTORUNS_EXE_PATH):
            safe_log(f"[+] Giải nén thành công vào: {AUTORUNS_DIR}")
            return True
        else:
            safe_log("[-] Giải nén thất bại, không tìm thấy Autoruns.exe.")
            return False
            
    except Exception as e:
        safe_log(f"[-] Lỗi trong quá trình tải/giải nén Autoruns: {e}")
        messagebox.showerror("Lỗi tải Autoruns", f"Không thể tải hoặc giải nén Autoruns. Vui lòng kiểm tra kết nối internet và thử lại.\nLỗi: {e}")
        return False

def scan_startup_items():
    """Chạy Autoruns để người dùng phân tích."""
    safe_log("🚀 Khởi chạy quét các mục khởi động...")
    if not download_and_unzip_autoruns():
        return
    
    try:
        safe_log(f"[+] Mở {AUTORUNS_EXE_PATH}...")
        # Sử dụng subprocess.Popen để không khóa giao diện chính
        subprocess.Popen([AUTORUNS_EXE_PATH])
        messagebox.showinfo("Hướng dẫn sử dụng Autoruns",
            "Autoruns đã được mở.\n\n"
            "💡 Mẹo phân tích:\n"
            "1. Vào 'Options' -> chọn 'Hide Microsoft Entries' và 'Hide Windows Entries' để ẩn các mục hệ thống.\n"
            "2. Chú ý đến các mục có màu HỒNG (không tìm thấy file) hoặc VÀNG (chưa được xác minh).\n"
            "3. Kiểm tra các tab 'Logon', 'Scheduled Tasks', và 'Services' để tìm các chương trình lạ.\n\n"
            "Bạn có thể bỏ dấu tick để vô hiệu hóa các mục đáng ngờ.")
    except Exception as e:
        safe_log(f"[-] Không thể chạy Autoruns: {e}")
        messagebox.showerror("Lỗi", f"Không thể khởi chạy Autoruns.exe.\nLỗi: {e}")

# ====== 9️⃣ GUI CALLBACKS ======
def start_detection():
    global running
    if not running:
        running = True
        threading.Thread(target=detect_loop, daemon=True).start()
        lbl_status.config(text="Trạng thái: Đang giám sát...", fg="green")

def stop_detection():
    global running
    running = False
    lbl_status.config(text="Trạng thái: Đã dừng", fg="red")
    safe_log("🔴 Đã dừng giám sát.")

# ====== 10️⃣ GUI MAIN ======
if __name__ == "__main__":
    ensure_admin()

    app = tk.Tk()
    app.title("Keylogger / Malware Detector (EMBER + Heuristic + Behavior)")
    app.geometry("720x560")  # Tăng chiều cao để có thêm không gian
    app.resizable(False, False)

    lbl_title = tk.Label(app, text="🔍 Keylogger / DLL Hook Detector", font=("Segoe UI", 15, "bold"))
    lbl_title.pack(pady=10)

    lbl_status = tk.Label(app, text="Trạng thái: Đã dừng", fg="red", font=("Segoe UI", 11))
    lbl_status.pack()

    # --- CẢI TIẾN: Thêm nút Quét Khởi động ---
    btn_frame = tk.Frame(app)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Bắt đầu Giám sát", width=18, command=start_detection, bg="#4CAF50", fg="white", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=5)
    tk.Button(btn_frame, text="Dừng Giám sát", width=18, command=stop_detection, bg="#F44336", fg="white", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, padx=5)
    tk.Button(btn_frame, text="Quét Mục Khởi Động", width=18, command=scan_startup_items, bg="#2196F3", fg="white", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5)


    lbl_info = tk.Label(app, text="Giám sát tiến trình thời gian thực và quét các chương trình tự khởi động cùng Windows.\n"
                                  "Lưu ý: Tiến trình hệ thống (Windows) được bỏ qua để tránh cảnh báo nhiễu.",
                                  wraplength=680, justify="center")
    lbl_info.pack(pady=5)

    txt_log = scrolledtext.ScrolledText(app, width=96, height=22, state="disabled", font=("Consolas", 9), bg="#2b2b2b", fg="#a9b7c6", insertbackground="white")
    txt_log.pack(padx=10, pady=10)

    load_model()
    app.mainloop()