#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import zipfile
import shutil
import logging
from pathlib import Path

# Setup logging với UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ember_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmberTrainer:
    """Class chính để training EMBER"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        # Thư mục source `ember/` nằm DƯỚI project root
        self.ember_dir = self.project_root / "ember"
        # Dataset nằm trong `data/ember2018` DƯỚI project root
        self.data_dir = self.project_root / "data" / "ember2018"
        self.model_path = self.project_root / "ember_model_pycharm.txt"
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Model path: {self.model_path}")
    
    def check_requirements(self):
        """Kiểm tra yêu cầu hệ thống"""
        logger.info("Kiem tra yeu cau he thong...")
        
        # Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"Python {python_version.major}.{python_version.minor} khong duoc ho tro. Can Python 3.8+")
            return False
        
        logger.info(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check available memory (approximate)
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            logger.info(f"💾 RAM: {memory_gb:.1f} GB")
            
            if memory_gb < 8:
                logger.warning("RAM thap (<8GB). Training co the cham hoac loi.")
            elif memory_gb >= 16:
                logger.info("RAM du cho training")
        except ImportError:
            logger.warning("Khong the kiem tra RAM. Cai dat psutil de kiem tra chi tiet.")
        
        return True
    
    def install_dependencies(self):
        """Cài đặt dependencies với progress tracking"""
        logger.info("Cai dat dependencies...")
        
        # Core packages
        packages = [
            "lightgbm",
            "tqdm", 
            "numpy",
            "pandas",
            "scikit-learn",
            "psutil"  # For system monitoring
        ]
        
        for package in packages:
            logger.info(f"Cai dat {package}...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package, "--quiet"
                ], check=True, capture_output=True)
                logger.info(f"{package} da cai dat")
            except subprocess.CalledProcessError as e:
                logger.error(f"Loi cai dat {package}: {e}")
                return False
        
        # Cài đặt LIEF (thử nhiều cách)
        logger.info("Cai dat LIEF...")
        lief_installed = False
        
        # Thử conda trước
        try:
            subprocess.run([
                "conda", "install", "-c", "conda-forge", "lief", "-y", "--quiet"
            ], check=True, capture_output=True)
            logger.info("LIEF da cai tu conda")
            lief_installed = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("Conda khong kha dung, thu pip...")
        
        # Thử pip nếu conda thất bại
        if not lief_installed:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "lief", "--quiet"
                ], check=True, capture_output=True)
                logger.info("LIEF da cai tu pip")
                lief_installed = True
            except subprocess.CalledProcessError as e:
                logger.error(f"Khong the cai LIEF: {e}")
                logger.error("Hay cai thu cong: conda install -c conda-forge lief")
                return False
        
        return True
    
    def setup_ember(self):
        """Setup EMBER repository"""
        logger.info("Setup EMBER repository...")
        
        if not self.ember_dir.exists():
            logger.info("Clone EMBER repository...")
            try:
                subprocess.run([
                    "git", "clone", "https://github.com/elastic/ember.git"
                ], check=True, cwd=self.project_root)
                logger.info("EMBER repository da clone")
            except subprocess.CalledProcessError as e:
                logger.error(f"Loi clone repository: {e}")
                return False
        else:
            logger.info("EMBER repository da co san")
        
        # Kiểm tra EMBER source code
        logger.info("Kiem tra EMBER source code...")
        if not (self.ember_dir / "__init__.py").exists():
            logger.error("Khong tim thay ember/__init__.py")
            return False
        
        if not (self.ember_dir / "features.py").exists():
            logger.error("Khong tim thay ember/features.py")
            return False
        
        logger.info("EMBER source code da co san")
        logger.info("Su dung EMBER truc tiep tu source code...")
        
        # Thêm project root vào sys.path để import `ember` từ source
        import sys
        project_root_str = str(self.project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            logger.info(f"Da them {project_root_str} vao sys.path")
        
        return True
    
    def setup_dataset(self):
        """Setup dataset"""
        logger.info("Setup dataset...")
        
        # Kiểm tra dataset có sẵn
        if self.data_dir.exists() and any(self.data_dir.glob("*.jsonl")):
            logger.info("Dataset da co san")
            logger.info(f"Dataset path: {self.data_dir}")
            # Liệt kê các file trong dataset
            files = list(self.data_dir.glob("*.jsonl"))
            logger.info(f"Tim thay {len(files)} file .jsonl")
            return True
        
        # Tìm file .zip
        zip_files = list(self.project_root.glob("*.zip"))
        if zip_files:
            logger.info(f"Tim thay file: {zip_files[0].name}")
            logger.info("Giai nen dataset...")
            
            try:
                with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                    zip_ref.extractall(self.project_root)
                logger.info("Dataset da duoc giai nen!")
                return True
            except Exception as e:
                logger.error(f"Loi giai nen: {e}")
                return False
        
        # Hướng dẫn upload dataset
        logger.error("Dataset chua co!")
        logger.info("Hay:")
        logger.info("1. Nen thu muc data/ember2018/ thanh file .zip")
        logger.info("2. Dat file .zip vao thu muc project")
        logger.info("3. Chay lai script")
        
        return False
    
    def train_ember(self):
        """Training EMBER model với monitoring"""
        logger.info("Bat dau training EMBER...")
        
        try:
            # Import EMBER từ source code
            import sys
            project_root_str = str(self.project_root)
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
            
            import ember
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            import lightgbm as lgb
            
            logger.info("EMBER da duoc import thanh cong")
        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.info("Thu import truc tiep tu source...")
            try:
                # Import trực tiếp từ source files
                import importlib.util
                spec = importlib.util.spec_from_file_location("ember", self.ember_dir / "__init__.py")
                ember = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ember)
                logger.info("EMBER da duoc import truc tiep")
            except Exception as e2:
                logger.error(f"Loi import truc tiep: {e2}")
                return None, None, None
        
        # Kiểm tra dataset đã có vectorized features chưa
        logger.info("Kiem tra dataset...")
        import json
        
        # Kiểm tra file metadata
        metadata_file = self.data_dir / "train_metadata.jsonl"
        if metadata_file.exists():
            logger.info("Tim thay train_metadata.jsonl")
            with open(metadata_file, 'r') as f:
                first_line = f.readline()
                logger.info(f"Metadata sample: {first_line[:100]}...")
        
        # Kiểm tra file features
        train_feature_files = sorted(list(self.data_dir.glob("train_features_*.jsonl")))
        test_feature_file = self.data_dir / "test_features.jsonl"
        
        if not train_feature_files:
            logger.error("Khong tim thay train_features_*.jsonl files")
            return None, None, None
        
        if not test_feature_file.exists():
            logger.error(f"Khong tim thay {test_feature_file.name}")
            logger.error("Dataset EMBER2018 can co file test_features.jsonl")
            return None, None, None
        
        logger.info(f"Tim thay {len(train_feature_files)} train feature files")
        logger.info(f"Tim thay {test_feature_file.name}")
        
        # Kiểm tra format file chi tiết
        logger.info("Kiem tra format file features...")
        try:
            with open(train_feature_files[0], 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    logger.error("File rong hoac khong co du lieu")
                    return None, None, None
                sample = json.loads(first_line)
                all_fields = list(sample.keys())
                logger.info(f"Fields trong sample: {all_fields}")
                
                if 'label' not in sample:
                    logger.error("File features KHONG co field 'label'!")
                    logger.error("EMBER2018 features files CAN CO field 'label' trong moi record.")
                    logger.error("Neu khong co, co the dataset bi sai hoac chua duoc xu ly dung.")
                    return None, None, None
                
                # Kiểm tra các field cần thiết cho feature extractor (EMBER dùng tên ngắn, không phải tên class)
                required_fields = ['histogram', 'byteentropy', 'general', 'header', 'section', 'imports', 'exports', 'strings']
                missing_fields = [f for f in required_fields if f not in sample]
                if missing_fields:
                    logger.error(f"Thieu cac fields bat buoc: {missing_fields}")
                    logger.error("Dataset EMBER2018 can co day du cac fields tren.")
                    logger.error("Neu thieu, dataset co the khong dung hoac chua duoc extract dung.")
                    return None, None, None
                
                # Kiểm tra format field 'section' - phải có 'entry' và 'sections'
                if 'section' in sample and isinstance(sample['section'], dict):
                    if 'entry' not in sample['section'] or 'sections' not in sample['section']:
                        logger.error("Field 'section' phai co 'entry' va 'sections'")
                        logger.error(f"Format hien tai: {list(sample['section'].keys())}")
                        return None, None, None
                    if not isinstance(sample['section']['entry'], str):
                        logger.error(f"Field 'section.entry' phai la string, nhung nhan duoc: {type(sample['section']['entry'])}")
                        return None, None, None
                
        except json.JSONDecodeError as e:
            logger.error(f"Loi parse JSON: {e}")
            logger.error("File features khong dung dinh dang JSON")
            return None, None, None
        except Exception as e:
            logger.error(f"Loi kiem tra format file: {e}")
            return None, None, None
        
        # Kiểm tra xem đã có vectorized features đầy đủ chưa (cần cả train và test)
        x_train_path = self.data_dir / "X_train.dat"
        y_train_path = self.data_dir / "y_train.dat"
        x_test_path = self.data_dir / "X_test.dat"
        y_test_path = self.data_dir / "y_test.dat"
        
        has_all_vectorized = (
            x_train_path.exists() and y_train_path.exists() and
            x_test_path.exists() and y_test_path.exists()
        )
        
        if has_all_vectorized:
            logger.info("Tim thay day du vectorized features (train + test), bo qua tao moi")
        else:
            missing = []
            if not x_train_path.exists(): missing.append("X_train.dat")
            if not y_train_path.exists(): missing.append("y_train.dat")
            if not x_test_path.exists(): missing.append("X_test.dat")
            if not y_test_path.exists(): missing.append("y_test.dat")
            logger.info(f"Thieu vectorized features: {', '.join(missing)}")
            logger.info("Tao vectorized features moi (co the mat 10-30 phut)...")
            logger.info("Dang vectorize tu JSONL files sang numpy arrays...")
            start_time = time.time()
            try:
                # Đảm bảo đường dẫn là string absolute
                data_dir_str = str(self.data_dir.absolute())
                logger.info(f"Data directory: {data_dir_str}")
                ember.create_vectorized_features(data_dir_str, feature_version=2)
                elapsed = time.time() - start_time
                logger.info(f"✓ Vectorized features hoan thanh trong {elapsed/60:.1f} phut ({elapsed:.2f} giay)")
            except Exception as e:
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
        
        # Ưu tiên dùng vectorized features (memory-mapped) của EMBER để tiết kiệm RAM
        logger.info("Loading vectorized features (memory-mapped)...")
        try:
            X_train, y_train, X_test, y_test = ember.read_vectorized_features(
                str(self.data_dir), feature_version=2
            )
            logger.info(f"Train: {X_train.shape[0]:,} samples")
            logger.info(f"Test: {X_test.shape[0]:,} samples")
        except Exception as e:
            logger.error(f"Khong the load vectorized features: {e}")
            logger.error("Hay dam bao da chay create_vectorized_features thanh cong hoac dataset dung dinh dang.")
            return None, None, None
        
        # LightGBM parameters (tối ưu RAM/CPU)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': 50,
            'lambda_l2': 0.1,
            'verbose': 0,
            'num_threads': max(1, (os.cpu_count() or 4) // 2),
            'force_col_wise': True  # Tối ưu cho dataset lớn
        }
        
        # Training với progress tracking
        logger.info("Training model...")
        logger.info("Thoi gian du kien: 30-60 phut")
        
        try:
            # Dùng Dataset trực tiếp từ mảng memory-mapped để giảm copy
            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, free_raw_data=False)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[test_data],
                num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(50), 
                    lgb.log_evaluation(100)
                ]
            )
            
            # Save model
            model.save_model(str(self.model_path))
            logger.info(f"Model da duoc luu: {self.model_path}")
            
            return model, X_test, y_test
            
        except Exception as e:
            logger.error(f"Loi training: {e}")
            return None, None, None
    
    def evaluate_model(self, model, X_test, y_test):
        """Đánh giá model với detailed metrics"""
        logger.info("Danh gia model...")
        
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, roc_auc_score, confusion_matrix,
                classification_report
            )
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary)
            recall = recall_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)
            auc = roc_auc_score(y_test, y_pred)
            
            logger.info("=" * 50)
            logger.info("KET QUA DANH GIA:")
            logger.info("=" * 50)
            logger.info(f"Accuracy:  {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall:    {recall:.4f}")
            logger.info(f"F1-Score:  {f1:.4f}")
            logger.info(f"AUC:       {auc:.4f}")
            logger.info("=" * 50)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_binary)
            logger.info("Confusion Matrix:")
            logger.info(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
            logger.info(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
            
            return model
            
        except Exception as e:
            logger.error(f"Loi danh gia: {e}")
            return model
    
    def test_model(self, model):
        """Test model với file mẫu"""
        logger.info("Test voi file mau...")
        
        try:
            import ember
            
            # Tạo file PE mẫu
            pe_header = b'MZ' + b'\x00' * 58 + b'PE\x00\x00' + b'\x00' * 1000
            test_file = self.project_root / 'test_sample.exe'
            
            with open(test_file, 'wb') as f:
                f.write(pe_header)
            
            score = ember.predict_sample(model, str(test_file), feature_version=2)
            
            logger.info("=" * 30)
            logger.info("KET QUA TEST:")
            logger.info("=" * 30)
            logger.info(f"Malware score: {score:.4f}")
            logger.info(f"Prediction: {'Malware' if score > 0.5 else 'Benign'}")
            logger.info("=" * 30)
            
            return model
            
        except Exception as e:
            logger.error(f"Loi test: {e}")
            return model
    
    def run_training(self):
        """Chạy toàn bộ quá trình training"""
        logger.info("=" * 60)
        logger.info("EMBER TRAINING CHO PYCHARM")
        logger.info("=" * 60)
        
        try:
            # 1. Kiểm tra yêu cầu
            if not self.check_requirements():
                return False
            
            # 2. Cài đặt dependencies
            if not self.install_dependencies():
                return False
            
            # 3. Setup EMBER
            if not self.setup_ember():
                return False
            
            # 4. Setup dataset
            if not self.setup_dataset():
                return False
            
            # 5. Training
            model, X_test, y_test = self.train_ember()
            if model is None:
                return False
            
            # 6. Đánh giá
            self.evaluate_model(model, X_test, y_test)
            
            # 7. Test
            self.test_model(model)
            
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

def main():
    """Hàm chính"""
    trainer = EmberTrainer()
    success = trainer.run_training()
    
    if success:
        print("\nTraining thanh cong!")
        print(f"Model: {trainer.model_path}")
        print("Log: ember_training.log")
    else:
        print("\nTraining that bai!")
        print("Xem log: ember_training.log")

if __name__ == "__main__":
    main()
