#!/usr/bin/env python3
"""
Script training EMBER Model (Version 1 - Chuẩn)
Mục đích: Train model LightGBM để phát hiện malware từ file PE (Windows executable)

Quy trình:
1. Kiểm tra yêu cầu hệ thống (Python, RAM)
2. Cài đặt dependencies (LightGBM, LIEF, numpy, pandas...)
3. Setup EMBER repository (clone hoặc dùng source có sẵn)
4. Setup dataset (kiểm tra hoặc giải nén từ .zip)
5. Train model LightGBM trên dataset EMBER2018
6. Đánh giá model (accuracy, precision, recall, F1, AUC)
7. Test model với file mẫu
"""

# ============================================================================
# PHẦN 1: IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ============================================================================
import os          # Thao tác với hệ thống file
import sys         # Thao tác với hệ thống Python (sys.path, version...)
import time        # Đo thời gian thực thi
import subprocess  # Chạy lệnh shell (pip install, git clone...)
import zipfile     # Giải nén file .zip (dataset)
import shutil      # Thao tác file/folder (copy, move...)
import logging     # Ghi log để theo dõi quá trình training
from pathlib import Path  # Xử lý đường dẫn file/folder dễ dàng hơn

# ============================================================================
# PHẦN 2: CẤU HÌNH LOGGING (GHI LOG)
# ============================================================================
# Logging giúp theo dõi quá trình training, ghi lại mọi thông tin quan trọng
logging.basicConfig(
    level=logging.INFO,  # Mức độ log: INFO (ghi tất cả thông tin quan trọng)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format: [Thời gian] - [Mức độ] - [Nội dung]
    handlers=[
        logging.FileHandler('ember_training.log', encoding='utf-8'),  # Ghi vào file log
        logging.StreamHandler()  # In ra màn hình console
    ]
)
logger = logging.getLogger(__name__)  # Tạo logger để dùng trong toàn bộ script

# ============================================================================
# PHẦN 3: CLASS EMBERTRAINER - CLASS CHÍNH ĐỂ TRAIN MODEL
# ============================================================================
class EmberTrainer:
    """
    Class chính để training EMBER model
    
    Class này chứa tất cả các phương thức cần thiết để:
    - Kiểm tra yêu cầu hệ thống
    - Cài đặt dependencies
    - Setup EMBER repository và dataset
    - Train model LightGBM
    - Đánh giá và test model
    """
    
    def __init__(self, project_root=None):
        """
        Khởi tạo EmberTrainer
        
        Args:
            project_root: Đường dẫn đến thư mục gốc của project (mặc định: thư mục hiện tại)
        """
        # Xác định thư mục gốc của project
        # Nếu không chỉ định, dùng thư mục hiện tại (nơi chạy script)
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Thư mục chứa source code EMBER (để import các hàm extract features)
        # Nằm trong project: project_root/ember/
        self.ember_dir = self.project_root / "ember"
        
        # Thư mục chứa dataset EMBER2018
        # Nằm trong project: project_root/data/ember2018/
        # Chứa các file: train_features_*.jsonl, test_features.jsonl, X_train.dat, y_train.dat...
        self.data_dir = self.project_root / "data" / "ember2018"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.model_path = self.project_root / f"{timestamp}_ember_model_pycharm.txt" 
        
        # Ghi log thông tin các đường dẫn quan trọng
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Model path: {self.model_path}")
    
    # ========================================================================
    # PHẦN 4: KIỂM TRA YÊU CẦU HỆ THỐNG
    # ========================================================================
    def check_requirements(self):
        """
        Kiểm tra yêu cầu hệ thống trước khi train
        
        Kiểm tra:
        - Phiên bản Python (cần >= 3.8)
        - RAM (khuyến nghị >= 8GB, tốt nhất >= 16GB)
        
        Returns:
            bool: True nếu đủ yêu cầu, False nếu thiếu
        """
        logger.info("Kiem tra yeu cau he thong...")
        
        # ====================================================================
        # Kiểm tra phiên bản Python
        # ====================================================================
        # EMBER cần Python 3.8+ để tương thích với các thư viện (numpy, pandas, lightgbm...)
        python_version = sys.version_info  # Lấy thông tin version: (major, minor, micro)
        if python_version < (3, 8):  # So sánh với (3, 8)
            logger.error(f"Python {python_version.major}.{python_version.minor} khong duoc ho tro. Can Python 3.8+")
            return False
        
        logger.info(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # ====================================================================
        # Kiểm tra RAM (bộ nhớ)
        # ====================================================================
        # Dataset EMBER2018 rất lớn (vài GB), cần đủ RAM để load vào memory
        try:
            import psutil  # Thư viện kiểm tra thông tin hệ thống
            memory = psutil.virtual_memory()  # Lấy thông tin RAM
            memory_gb = memory.total / (1024**3)  # Chuyển từ bytes sang GB
            logger.info(f"💾 RAM: {memory_gb:.1f} GB")
            
            # Đánh giá RAM
            if memory_gb < 8:
                # RAM < 8GB: Có thể chậm hoặc lỗi out of memory
                logger.warning("RAM thap (<8GB). Training co the cham hoac loi.")
            elif memory_gb >= 16:
                # RAM >= 16GB: Đủ cho training mượt mà
                logger.info("RAM du cho training")
        except ImportError:
            # Nếu chưa cài psutil, bỏ qua (không bắt buộc)
            logger.warning("Khong the kiem tra RAM. Cai dat psutil de kiem tra chi tiet.")
        
        return True
    
    # ========================================================================
    # PHẦN 5: CÀI ĐẶT DEPENDENCIES (CÁC THƯ VIỆN CẦN THIẾT)
    # ========================================================================
    def install_dependencies(self):
        """
        Tự động cài đặt các thư viện Python cần thiết cho training
        
        Cài đặt:
        - lightgbm: Thuật toán machine learning (gradient boosting)
        - numpy, pandas: Xử lý dữ liệu số và bảng
        - scikit-learn: Metrics đánh giá model
        - tqdm: Hiển thị progress bar
        - psutil: Kiểm tra RAM/CPU
        - lief: Parse file PE (Portable Executable) để extract features
        
        Returns:
            bool: True nếu cài đặt thành công, False nếu lỗi
        """
        logger.info("Cai dat dependencies...")
        
        # ====================================================================
        # Cài đặt các package cơ bản (dễ cài, dùng pip)
        # ====================================================================
        packages = [
            "lightgbm",      # Thuật toán ML chính (Light Gradient Boosting Machine)
            "tqdm",          # Progress bar (hiển thị tiến trình)
            "numpy",         # Xử lý mảng số (arrays)
            "pandas",        # Xử lý dữ liệu dạng bảng (DataFrame)
            "scikit-learn",  # Metrics đánh giá (accuracy, precision, recall...)
            "psutil"         # Kiểm tra thông tin hệ thống (RAM, CPU)
        ]
        
        # Cài từng package một
        for package in packages:
            logger.info(f"Cai dat {package}...")
            try:
                # Chạy lệnh: python -m pip install <package> --quiet
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package, "--quiet"
                ], check=True, capture_output=True)
                logger.info(f"{package} da cai dat")
            except subprocess.CalledProcessError as e:
                logger.error(f"Loi cai dat {package}: {e}")
                return False
        
        # ====================================================================
        # Cài đặt LIEF (phức tạp hơn, thử nhiều cách)
        # ====================================================================
        # LIEF (Library to Instrument Executable Formats) dùng để parse file PE
        # Cần để extract features từ file .exe, .dll...
        logger.info("Cai dat LIEF...")
        lief_installed = False
        
        # Cách 1: Thử cài bằng conda (thường ổn định hơn cho LIEF)
        try:
            subprocess.run([
                "conda", "install", "-c", "conda-forge", "lief", "-y", "--quiet"
            ], check=True, capture_output=True)
            logger.info("LIEF da cai tu conda")
            lief_installed = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Conda không có sẵn hoặc lỗi → thử pip
            logger.info("Conda khong kha dung, thu pip...")
        
        # Cách 2: Thử cài bằng pip (nếu conda thất bại)
        if not lief_installed:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "lief", "--quiet"
                ], check=True, capture_output=True)
                logger.info("LIEF da cai tu pip")
                lief_installed = True
            except subprocess.CalledProcessError as e:
                # Cả 2 cách đều thất bại → hướng dẫn cài thủ công
                logger.error(f"Khong the cai LIEF: {e}")
                logger.error("Hay cai thu cong: conda install -c conda-forge lief")
                return False
        
        return True
    
    # ========================================================================
    # PHẦN 6: SETUP EMBER REPOSITORY (SOURCE CODE)
    # ========================================================================
    def setup_ember(self):
        """
        Setup EMBER repository (source code)
        
        EMBER là một Python package chứa các hàm:
        - extract features từ file PE
        - create vectorized features từ JSONL
        - train model LightGBM
        
        Nếu chưa có, sẽ clone từ GitHub.
        Nếu đã có, sẽ dùng source code có sẵn.
        
        Returns:
            bool: True nếu setup thành công, False nếu lỗi
        """
        logger.info("Setup EMBER repository...")
        
        # ====================================================================
        # Kiểm tra xem đã có thư mục ember/ chưa
        # ====================================================================
        if not self.ember_dir.exists():
            # Chưa có → clone từ GitHub
            logger.info("Clone EMBER repository...")
            try:
                # Chạy lệnh: git clone https://github.com/elastic/ember.git
                subprocess.run([
                    "git", "clone", "https://github.com/elastic/ember.git"
                ], check=True, cwd=self.project_root)
                logger.info("EMBER repository da clone")
            except subprocess.CalledProcessError as e:
                logger.error(f"Loi clone repository: {e}")
                return False
        else:
            # Đã có → dùng source code có sẵn
            logger.info("EMBER repository da co san")
        
        # ====================================================================
        # Kiểm tra các file quan trọng trong EMBER
        # ====================================================================
        logger.info("Kiem tra EMBER source code...")
        
        # File __init__.py: Để Python nhận diện đây là package
        if not (self.ember_dir / "__init__.py").exists():
            logger.error("Khong tim thay ember/__init__.py")
            return False
        
        # File features.py: Chứa code extract features từ PE file
        if not (self.ember_dir / "features.py").exists():
            logger.error("Khong tim thay ember/features.py")
            return False
        
        logger.info("EMBER source code da co san")
        logger.info("Su dung EMBER truc tiep tu source code...")
        
        # ====================================================================
        # Thêm project root vào sys.path để Python có thể import ember
        # ====================================================================
        # sys.path là danh sách các thư mục Python tìm kiếm khi import
        # Thêm project_root vào đầu danh sách để ưu tiên import từ source code
        import sys
        project_root_str = str(self.project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)  # Thêm vào đầu (ưu tiên cao nhất)
            logger.info(f"Da them {project_root_str} vao sys.path")
        
        return True
    
    # ========================================================================
    # PHẦN 7: SETUP DATASET (DỮ LIỆU TRAINING)
    # ========================================================================
    def setup_dataset(self):
        """
        Setup dataset EMBER2018
        
        Dataset EMBER2018 chứa:
        - train_features_0.jsonl đến train_features_5.jsonl (6 files)
        - test_features.jsonl
        - Mỗi file chứa features đã extract từ file PE (malware/benign)
        
        Nếu chưa có, sẽ tìm file .zip để giải nén.
        
        Returns:
            bool: True nếu dataset có sẵn, False nếu thiếu
        """
        logger.info("Setup dataset...")
        
        # ====================================================================
        # Kiểm tra xem dataset đã có sẵn chưa
        # ====================================================================
        # Dataset nằm trong data/ember2018/ và có các file .jsonl
        if self.data_dir.exists() and any(self.data_dir.glob("*.jsonl")):
            logger.info("Dataset da co san")
            logger.info(f"Dataset path: {self.data_dir}")
            # Liệt kê các file .jsonl tìm thấy
            files = list(self.data_dir.glob("*.jsonl"))
            logger.info(f"Tim thay {len(files)} file .jsonl")
            return True
        
        # ====================================================================
        # Nếu chưa có, tìm file .zip để giải nén
        # ====================================================================
        zip_files = list(self.project_root.glob("*.zip"))
        if zip_files:
            logger.info(f"Tim thay file: {zip_files[0].name}")
            logger.info("Giai nen dataset...")
            
            try:
                # Giải nén file .zip vào thư mục project
                with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                    zip_ref.extractall(self.project_root)
                logger.info("Dataset da duoc giai nen!")
                return True
            except Exception as e:
                logger.error(f"Loi giai nen: {e}")
                return False
        
        # ====================================================================
        # Nếu không tìm thấy dataset → hướng dẫn người dùng
        # ====================================================================
        logger.error("Dataset chua co!")
        logger.info("Hay:")
        logger.info("1. Nen thu muc data/ember2018/ thanh file .zip")
        logger.info("2. Dat file .zip vao thu muc project")
        logger.info("3. Chay lai script")
        
        return False
    
    # ========================================================================
    # PHẦN 8: TRAIN MODEL EMBER (PHẦN QUAN TRỌNG NHẤT)
    # ========================================================================
    def train_ember(self):
        """
        Training EMBER model với LightGBM
        
        Quy trình:
        1. Import các thư viện cần thiết (ember, lightgbm, numpy...)
        2. Kiểm tra dataset (JSONL files)
        3. Tạo vectorized features (nếu chưa có) - chuyển JSONL → numpy arrays
        4. Load vectorized features (memory-mapped để tiết kiệm RAM)
        5. Train LightGBM model
        6. Lưu model vào file .txt
        
        Returns:
            tuple: (model, X_test, y_test) nếu thành công, (None, None, None) nếu lỗi
        """
        logger.info("Bat dau training EMBER...")
        
        # ====================================================================
        # BƯỚC 1: IMPORT CÁC THƯ VIỆN CẦN THIẾT
        # ====================================================================
        try:
            # Đảm bảo có thể import ember từ source code
            import sys
            project_root_str = str(self.project_root)
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
            
            # Import các thư viện chính
            import ember          # EMBER package (extract features, train model...)
            import numpy as np    # Xử lý mảng số
            import pandas as pd   # Xử lý dữ liệu dạng bảng
            from sklearn.model_selection import train_test_split  # Chia train/test (không dùng ở đây, dataset đã chia sẵn)
            import lightgbm as lgb  # Thuật toán ML chính (LightGBM)
            
            logger.info("EMBER da duoc import thanh cong")
        except ImportError as e:
            # Nếu import thất bại → thử import trực tiếp từ file
            logger.error(f"Import error: {e}")
            logger.info("Thu import truc tiep tu source...")
            try:
                # Import trực tiếp từ file __init__.py
                import importlib.util
                spec = importlib.util.spec_from_file_location("ember", self.ember_dir / "__init__.py")
                ember = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ember)
                logger.info("EMBER da duoc import truc tiep")
            except Exception as e2:
                logger.error(f"Loi import truc tiep: {e2}")
                return None, None, None
        
        # ====================================================================
        # BƯỚC 2: KIỂM TRA DATASET (CÁC FILE JSONL)
        # ====================================================================
        logger.info("Kiem tra dataset...")
        import json
        
        # Kiểm tra file metadata (không bắt buộc, chỉ để tham khảo)
        metadata_file = self.data_dir / "train_metadata.jsonl"
        if metadata_file.exists():
            logger.info("Tim thay train_metadata.jsonl")
            with open(metadata_file, 'r') as f:
                first_line = f.readline()
                logger.info(f"Metadata sample: {first_line[:100]}...")
        
        # ====================================================================
        # Kiểm tra các file features chính (BẮT BUỘC)
        # ====================================================================
        # Tìm các file train_features_*.jsonl (train_features_0.jsonl đến train_features_5.jsonl)
        train_feature_files = sorted(list(self.data_dir.glob("train_features_*.jsonl")))
        # Tìm file test_features.jsonl
        test_feature_file = self.data_dir / "test_features.jsonl"
        
        # Kiểm tra xem có đủ file không
        if not train_feature_files:
            logger.error("Khong tim thay train_features_*.jsonl files")
            return None, None, None
        
        if not test_feature_file.exists():
            logger.error(f"Khong tim thay {test_feature_file.name}")
            logger.error("Dataset EMBER2018 can co file test_features.jsonl")
            return None, None, None
        
        logger.info(f"Tim thay {len(train_feature_files)} train feature files")
        logger.info(f"Tim thay {test_feature_file.name}")
        
        # ====================================================================
        # BƯỚC 3: KIỂM TRA FORMAT FILE FEATURES (ĐỊNH DẠNG JSONL)
        # ====================================================================
        # Mỗi dòng trong file .jsonl là một JSON object chứa features của 1 file PE
        logger.info("Kiem tra format file features...")
        try:
            # Đọc dòng đầu tiên để kiểm tra format
            with open(train_feature_files[0], 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    logger.error("File rong hoac khong co du lieu")
                    return None, None, None
                
                # Parse JSON từ dòng đầu tiên
                sample = json.loads(first_line)
                all_fields = list(sample.keys())
                logger.info(f"Fields trong sample: {all_fields}")
                
                # ============================================================
                # Kiểm tra field 'label' (BẮT BUỘC)
                # ============================================================
                # label = 0 (benign) hoặc 1 (malware)
                if 'label' not in sample:
                    logger.error("File features KHONG co field 'label'!")
                    logger.error("EMBER2018 features files CAN CO field 'label' trong moi record.")
                    logger.error("Neu khong co, co the dataset bi sai hoac chua duoc xu ly dung.")
                    return None, None, None
                
                # ============================================================
                # Kiểm tra các field features cần thiết (BẮT BUỘC)
                # ============================================================
                # EMBER extract 9 nhóm features:
                # - histogram: Byte histogram (256 features)
                # - byteentropy: Byte entropy histogram (256 features)
                # - general: General file info (10 features)
                # - header: PE header info (62 features)
                # - section: Section info (319 features)
                # - imports: Imported functions (1280 features)
                # - exports: Exported functions (128 features)
                # - strings: String features (68 features)
                required_fields = ['histogram', 'byteentropy', 'general', 'header', 'section', 'imports', 'exports', 'strings']
                missing_fields = [f for f in required_fields if f not in sample]
                if missing_fields:
                    logger.error(f"Thieu cac fields bat buoc: {missing_fields}")
                    logger.error("Dataset EMBER2018 can co day du cac fields tren.")
                    logger.error("Neu thieu, dataset co the khong dung hoac chua duoc extract dung.")
                    return None, None, None
                
                # ============================================================
                # Kiểm tra format field 'section' (chi tiết)
                # ============================================================
                # Field 'section' phải là dict có 'entry' và 'sections'
                if 'section' in sample and isinstance(sample['section'], dict):
                    if 'entry' not in sample['section'] or 'sections' not in sample['section']:
                        logger.error("Field 'section' phai co 'entry' va 'sections'")
                        logger.error(f"Format hien tai: {list(sample['section'].keys())}")
                        return None, None, None
                    if not isinstance(sample['section']['entry'], str):
                        logger.error(f"Field 'section.entry' phai la string, nhung nhan duoc: {type(sample['section']['entry'])}")
                        return None, None, None
                
        except json.JSONDecodeError as e:
            # Lỗi parse JSON → file không đúng định dạng
            logger.error(f"Loi parse JSON: {e}")
            logger.error("File features khong dung dinh dang JSON")
            return None, None, None
        except Exception as e:
            logger.error(f"Loi kiem tra format file: {e}")
            return None, None, None
        
        # ====================================================================
        # BƯỚC 4: TẠO VECTORIZED FEATURES (NẾU CHƯA CÓ)
        # ====================================================================
        # Vectorized features = Chuyển JSONL → numpy arrays (.dat files)
        # Lý do: JSONL chậm, numpy arrays nhanh hơn và tiết kiệm RAM (memory-mapped)
        
        # Kiểm tra xem đã có vectorized features đầy đủ chưa (cần cả train và test)
        x_train_path = self.data_dir / "X_train.dat"  # Features training (numpy array)
        y_train_path = self.data_dir / "y_train.dat"   # Labels training (0/1)
        x_test_path = self.data_dir / "X_test.dat"     # Features test (numpy array)
        y_test_path = self.data_dir / "y_test.dat"      # Labels test (0/1)
        
        has_all_vectorized = (
            x_train_path.exists() and y_train_path.exists() and
            x_test_path.exists() and y_test_path.exists()
        )
        
        if has_all_vectorized:
            # Đã có đầy đủ → bỏ qua bước tạo mới (tiết kiệm thời gian)
            logger.info("Tim thay day du vectorized features (train + test), bo qua tao moi")
        else:
            # Thiếu một số file → cần tạo mới
            missing = []
            if not x_train_path.exists(): missing.append("X_train.dat")
            if not y_train_path.exists(): missing.append("y_train.dat")
            if not x_test_path.exists(): missing.append("X_test.dat")
            if not y_test_path.exists(): missing.append("y_test.dat")
            logger.info(f"Thieu vectorized features: {', '.join(missing)}")
            logger.info("Tao vectorized features moi (co the mat 10-30 phut)...")
            logger.info("Dang vectorize tu JSONL files sang numpy arrays...")
            
            # Đo thời gian tạo features
            start_time = time.time()
            try:
                # Gọi hàm EMBER để tạo vectorized features
                # Hàm này sẽ:
                # 1. Đọc tất cả file train_features_*.jsonl và test_features.jsonl
                # 2. Extract features từ mỗi record JSON
                # 3. Chuyển thành numpy arrays (2381 features mỗi sample)
                # 4. Lưu vào file .dat (memory-mapped format)
                data_dir_str = str(self.data_dir.absolute())
                logger.info(f"Data directory: {data_dir_str}")
                ember.create_vectorized_features(data_dir_str, feature_version=2)
                
                elapsed = time.time() - start_time
                logger.info(f"✓ Vectorized features hoan thanh trong {elapsed/60:.1f} phut ({elapsed:.2f} giay)")
            except Exception as e:
                # Lỗi khi tạo features → hướng dẫn debug
                logger.error(f"✗ Loi tao features: {e}")
                logger.error(f"Chi tiet loi: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                logger.error("\nKiem tra:")
                logger.error(f"  1. File train_features_0.jsonl den train_features_5.jsonl co ton tai khong?")
                logger.error(f"  2. File test_features.jsonl co ton tai khong?")
                logger.error(f"  3. Cac file JSONL co dung dinh dang EMBER2018 khong?")
                logger.error(f"  4. Co du RAM va disk space khong?")
                return None, None, None
        
        # ====================================================================
        # BƯỚC 5: LOAD VECTORIZED FEATURES (MEMORY-MAPPED)
        # ====================================================================
        # Memory-mapped = Không load toàn bộ vào RAM, chỉ map vào memory
        # Lợi ích: Tiết kiệm RAM (dataset rất lớn, vài GB)
        logger.info("Loading vectorized features (memory-mapped)...")
        try:
            # Hàm này load 4 file .dat:
            # - X_train.dat: Features training (shape: [n_samples, 2381])
            # - y_train.dat: Labels training (shape: [n_samples]) - 0 hoặc 1
            # - X_test.dat: Features test (shape: [n_samples, 2381])
            # - y_test.dat: Labels test (shape: [n_samples]) - 0 hoặc 1
            X_train, y_train, X_test, y_test = ember.read_vectorized_features(
                str(self.data_dir), feature_version=2
            )
            logger.info(f"Train: {X_train.shape[0]:,} samples x {X_train.shape[1]:,} features")
            logger.info(f"Test: {X_test.shape[0]:,} samples x {X_test.shape[1]:,} features")
        except Exception as e:
            logger.error(f"Khong the load vectorized features: {e}")
            logger.error("Hay dam bao da chay create_vectorized_features thanh cong hoac dataset dung dinh dang.")
            return None, None, None
        
        # ====================================================================
        # BƯỚC 6: CẤU HÌNH LIGHTGBM PARAMETERS
        # ====================================================================
        # LightGBM là thuật toán gradient boosting, cần cấu hình các tham số
        params = {
            'objective': 'binary',        # Binary classification (malware/benign)
            'metric': 'auc',              # Metric đánh giá: AUC (Area Under Curve)
            'boosting_type': 'gbdt',      # Gradient Boosting Decision Tree
            'num_leaves': 31,             # Số lá trong mỗi cây (nhỏ → nhanh, ít overfit)
            'learning_rate': 0.05,         # Tốc độ học (nhỏ → chậm nhưng chính xác hơn)
            'feature_fraction': 0.8,       # Dùng 80% features mỗi cây (giảm overfit)
            'bagging_fraction': 0.8,       # Dùng 80% samples mỗi cây (giảm overfit)
            'bagging_freq': 1,             # Bagging mỗi 1 iteration
            'min_data_in_leaf': 50,        # Tối thiểu 50 samples mỗi lá (tránh overfit)
            'lambda_l2': 0.1,              # L2 regularization (giảm overfit)
            'verbose': 0,                 # Không in log chi tiết (dùng callback thay thế)
            'num_threads': max(1, (os.cpu_count() or 4) // 2),  # Dùng 50% CPU cores
            'force_col_wise': True         # Tối ưu cho dataset lớn (theo cột)
        }
        
        # ====================================================================
        # BƯỚC 7: TRAIN MODEL LIGHTGBM
        # ====================================================================
        logger.info("Training model...")
        logger.info("Thoi gian du kien: 30-60 phut")
        
        try:
            # Tạo LightGBM Dataset từ numpy arrays
            # free_raw_data=False: Giữ nguyên data gốc (memory-mapped), không copy
            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)
            
            # Train model
            model = lgb.train(
                params,                    # Parameters đã cấu hình
                train_data,                # Data training
                valid_sets=[test_data],    # Validation set (để early stopping)
                num_boost_round=500,       # Tối đa 500 cây (có thể dừng sớm nếu early stopping)
                callbacks=[
                    lgb.early_stopping(50),   # Dừng nếu không cải thiện sau 50 rounds
                    lgb.log_evaluation(100)    # In log mỗi 100 rounds
                ]
            )
            
            # ================================================================
            # BƯỚC 8: LƯU MODEL
            # ================================================================
            # Lưu model vào file .txt (LightGBM text format, human-readable)
            model.save_model(str(self.model_path))
            logger.info(f"Model da duoc luu: {self.model_path}")
            
            return model, X_test, y_test
            
        except Exception as e:
            logger.error(f"Loi training: {e}")
            return None, None, None
    
    # ========================================================================
    # PHẦN 9: ĐÁNH GIÁ MODEL (EVALUATION)
    # ========================================================================
    def evaluate_model(self, model, X_test, y_test):
        """
        Đánh giá model với các metrics chi tiết
        
        Metrics được tính:
        - Accuracy: Độ chính xác tổng thể
        - Precision: Độ chính xác dương (trong số dự đoán malware, bao nhiêu % đúng)
        - Recall: Độ nhạy (trong số malware thực tế, bao nhiêu % được phát hiện)
        - F1-Score: Cân bằng giữa Precision và Recall
        - AUC: Khả năng phân biệt malware/benign
        
        Args:
            model: LightGBM model đã train
            X_test: Test features (numpy array)
            y_test: Test labels (0 hoặc 1)
        
        Returns:
            model: Model đã được đánh giá
        """
        logger.info("Danh gia model...")
        
        try:
            # Import các hàm tính metrics từ scikit-learn
            from sklearn.metrics import (
                accuracy_score,    # Độ chính xác
                precision_score,   # Precision
                recall_score,      # Recall
                f1_score,          # F1-Score
                roc_auc_score,     # AUC
                confusion_matrix,   # Confusion Matrix (TN, FP, FN, TP)
                classification_report
            )
            
            # ================================================================
            # Dự đoán trên test set
            # ================================================================
            # y_pred: Xác suất malware (0.0 - 1.0)
            y_pred = model.predict(X_test)
            # y_pred_binary: Dự đoán nhị phân (0 hoặc 1) - threshold = 0.5
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # ================================================================
            # Tính các metrics
            # ================================================================
            accuracy = accuracy_score(y_test, y_pred_binary)   # (TP + TN) / (TP + TN + FP + FN)
            precision = precision_score(y_test, y_pred_binary)  # TP / (TP + FP)
            recall = recall_score(y_test, y_pred_binary)        # TP / (TP + FN)
            f1 = f1_score(y_test, y_pred_binary)               # 2 * (precision * recall) / (precision + recall)
            auc = roc_auc_score(y_test, y_pred)                 # Area Under ROC Curve
            
            # In kết quả
            logger.info("=" * 50)
            logger.info("KET QUA DANH GIA:")
            logger.info("=" * 50)
            logger.info(f"Accuracy:  {accuracy:.4f}")   # Ví dụ: 0.9450 = 94.5%
            logger.info(f"Precision: {precision:.4f}")  # Ví dụ: 0.9800 = 98.0%
            logger.info(f"Recall:    {recall:.4f}")     # Ví dụ: 0.9000 = 90.0%
            logger.info(f"F1-Score:  {f1:.4f}")         # Ví dụ: 0.9380 = 93.8%
            logger.info(f"AUC:       {auc:.4f}")        # Ví dụ: 0.9900 = 99.0%
            logger.info("=" * 50)
            
            # ================================================================
            # Confusion Matrix (Ma trận nhầm lẫn)
            # ================================================================
            # TN (True Negative):  Dự đoán benign, thực tế benign (đúng)
            # FP (False Positive): Dự đoán malware, thực tế benign (sai - báo động sai)
            # FN (False Negative): Dự đoán benign, thực tế malware (sai - bỏ sót malware)
            # TP (True Positive):  Dự đoán malware, thực tế malware (đúng)
            cm = confusion_matrix(y_test, y_pred_binary)
            logger.info("Confusion Matrix:")
            logger.info(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")  # Hàng 1: Benign thực tế
            logger.info(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")  # Hàng 2: Malware thực tế
            
            return model
            
        except Exception as e:
            logger.error(f"Loi danh gia: {e}")
            return model
    
    # ========================================================================
    # PHẦN 10: TEST MODEL VỚI FILE MẪU
    # ========================================================================
    def test_model(self, model):
        """
        Test model với file PE mẫu (tự tạo)
        
        Tạo một file PE giả (chỉ có header MZ và PE) để test xem model có hoạt động không.
        File này không phải malware thật, chỉ để kiểm tra pipeline.
        
        Args:
            model: LightGBM model đã train
        
        Returns:
            model: Model đã được test
        """
        logger.info("Test voi file mau...")
        
        try:
            import ember
            
            # ================================================================
            # Tạo file PE mẫu (giả)
            # ================================================================
            # File PE hợp lệ phải bắt đầu với:
            # - "MZ" (2 bytes đầu) - DOS header
            # - "PE\x00\x00" (ở offset 0x3C) - PE signature
            pe_header = b'MZ' + b'\x00' * 58 + b'PE\x00\x00' + b'\x00' * 1000
            test_file = self.project_root / 'test_sample.exe'
            
            # Ghi file PE mẫu
            with open(test_file, 'wb') as f:
                f.write(pe_header)
            
            # ================================================================
            # Dự đoán với file mẫu
            # ================================================================
            # Hàm predict_sample cần file_data (bytes), không phải file path
            # Quy trình:
            # 1. Đọc file PE dưới dạng binary (bytes)
            # 2. Extract features (2381 features) từ bytes
            # 3. Predict với model
            # 4. Trả về score (0.0 - 1.0)
            with open(test_file, 'rb') as f:
                file_data = f.read()
            
            score = ember.predict_sample(model, file_data, feature_version=2)
            
            # In kết quả
            logger.info("=" * 30)
            logger.info("KET QUA TEST:")
            logger.info("=" * 30)
            logger.info(f"Malware score: {score:.4f}")  # Ví dụ: 0.1234
            logger.info(f"Prediction: {'Malware' if score > 0.5 else 'Benign'}")  # > 0.5 = Malware
            logger.info("=" * 30)
            
            return model
            
        except Exception as e:
            logger.error(f"Loi test: {e}")
            return model
    
    # ========================================================================
    # PHẦN 11: CHẠY TOÀN BỘ QUÁ TRÌNH TRAINING
    # ========================================================================
    def run_training(self):
        """
        Chạy toàn bộ quá trình training từ đầu đến cuối
        
        Quy trình:
        1. Kiểm tra yêu cầu hệ thống (Python, RAM)
        2. Cài đặt dependencies (LightGBM, LIEF, numpy...)
        3. Setup EMBER repository (clone hoặc dùng source có sẵn)
        4. Setup dataset (kiểm tra hoặc giải nén)
        5. Training model LightGBM
        6. Đánh giá model (metrics)
        7. Test model với file mẫu
        
        Returns:
            bool: True nếu thành công, False nếu lỗi
        """
        logger.info("=" * 60)
        logger.info("EMBER TRAINING CHO PYCHARM")
        logger.info("=" * 60)
        
        try:
            # ================================================================
            # BƯỚC 1: Kiểm tra yêu cầu hệ thống
            # ================================================================
            if not self.check_requirements():
                return False
            
            # ================================================================
            # BƯỚC 2: Cài đặt dependencies
            # ================================================================
            if not self.install_dependencies():
                return False
            
            # ================================================================
            # BƯỚC 3: Setup EMBER repository
            # ================================================================
            if not self.setup_ember():
                return False
            
            # ================================================================
            # BƯỚC 4: Setup dataset
            # ================================================================
            if not self.setup_dataset():
                return False
            
            # ================================================================
            # BƯỚC 5: Training model
            # ================================================================
            model, X_test, y_test = self.train_ember()
            if model is None:
                return False
            
            # ================================================================
            # BƯỚC 6: Đánh giá model
            # ================================================================
            self.evaluate_model(model, X_test, y_test)
            
            # ================================================================
            # BƯỚC 7: Test model với file mẫu
            # ================================================================
            self.test_model(model)
            
            # ================================================================
            # Hoàn tất - In thông tin model
            # ================================================================
            logger.info("=" * 60)
            logger.info("TRAINING HOAN TAT!")
            logger.info("=" * 60)
            logger.info(f"Model: {self.model_path}")
            logger.info("Su dung model:")
            logger.info("   import lightgbm as lgb")
            logger.info(f"   model = lgb.Booster(model_file='{self.model_path}')")
            
            return True
            
        except Exception as e:
            logger.error(f"Loi tong quat: {e}")
            return False

# ============================================================================
# PHẦN 12: HÀM MAIN - ĐIỂM VÀO CHÍNH CỦA SCRIPT
# ============================================================================
def main():
    """
    Hàm chính - Điểm vào của script
    
    Khi chạy script: python train/ember_v1.py
    Hàm này sẽ:
    1. Tạo đối tượng EmberTrainer
    2. Chạy toàn bộ quá trình training
    3. In kết quả (thành công hoặc thất bại)
    """
    # Tạo đối tượng trainer
    trainer = EmberTrainer()
    
    # Chạy toàn bộ quá trình training
    success = trainer.run_training()
    
    # In kết quả
    if success:
        print("\nTraining thanh cong!")
        print(f"Model: {trainer.model_path}")
        print("Log: ember_training.log")
    else:
        print("\nTraining that bai!")
        print("Xem log: ember_training.log")

# ============================================================================
# PHẦN 13: CHẠY SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Chỉ chạy hàm main() khi script được gọi trực tiếp
    # (không chạy khi được import như module)
    main()
