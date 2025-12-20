#!/usr/bin/env python3
"""
Script để so sánh 2 model EMBER với nhau
So sánh: số cây, số features, kích thước, và hiệu năng trên test set
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_comparison.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Class để so sánh 2 model EMBER"""
    
    def __init__(self, model1_path, model2_path, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.model1_path = self._resolve_path(model1_path)
        self.model2_path = self._resolve_path(model2_path)
        
        # Setup import path
        project_root_str = str(self.project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        
        # Load models
        self.model1 = None
        self.model2 = None
        self.load_models()
    
    def _resolve_path(self, model_path):
        """Resolve model path (absolute or relative)"""
        path = Path(model_path)
        if not path.is_absolute():
            path = self.project_root / path
        if not path.exists():
            raise FileNotFoundError(f"Model file không tồn tại: {path}")
        return path
    
    def load_models(self):
        """Load cả 2 model"""
        try:
            import lightgbm as lgb
            
            logger.info("=" * 80)
            logger.info("ĐANG LOAD 2 MODEL ĐỂ SO SÁNH")
            logger.info("=" * 80)
            
            # Load model 1
            logger.info(f"Đang load model 1: {self.model1_path.name}")
            self.model1 = lgb.Booster(model_file=str(self.model1_path))
            num_trees1 = self.model1.num_trees()
            num_features1 = self.model1.num_feature()
            size1_mb = self.model1_path.stat().st_size / (1024**2)
            logger.info(f"✓ Model 1 đã load: {num_trees1:,} cây, {num_features1:,} features, {size1_mb:.1f} MB")
            
            # Load model 2
            logger.info(f"Đang load model 2: {self.model2_path.name}")
            self.model2 = lgb.Booster(model_file=str(self.model2_path))
            num_trees2 = self.model2.num_trees()
            num_features2 = self.model2.num_feature()
            size2_mb = self.model2_path.stat().st_size / (1024**2)
            logger.info(f"✓ Model 2 đã load: {num_trees2:,} cây, {num_features2:,} features, {size2_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"Lỗi load model: {e}")
            raise
    
    def compare_basic_info(self):
        """So sánh thông tin cơ bản của 2 model"""
        logger.info("=" * 80)
        logger.info("SO SÁNH THÔNG TIN CƠ BẢN")
        logger.info("=" * 80)
        
        num_trees1 = self.model1.num_trees()
        num_features1 = self.model1.num_feature()
        size1_mb = self.model1_path.stat().st_size / (1024**2)
        
        num_trees2 = self.model2.num_trees()
        num_features2 = self.model2.num_feature()
        size2_mb = self.model2_path.stat().st_size / (1024**2)
        
        logger.info(f"{'Tiêu chí':<30} | {'Model 1':<25} | {'Model 2':<25} | {'Khác biệt'}")
        logger.info("-" * 80)
        logger.info(f"{'Tên file':<30} | {self.model1_path.name:<25} | {self.model2_path.name:<25} |")
        logger.info(f"{'Số cây':<30} | {num_trees1:>25,} | {num_trees2:>25,} | {num_trees2 - num_trees1:+,}")
        logger.info(f"{'Số features':<30} | {num_features1:>25,} | {num_features2:>25,} | {num_features2 - num_features1:+,}")
        logger.info(f"{'Kích thước (MB)':<30} | {size1_mb:>25.1f} | {size2_mb:>25.1f} | {size2_mb - size1_mb:+.1f} MB")
        
        # Đánh giá
        logger.info("")
        logger.info("📊 ĐÁNH GIÁ:")
        
        # Model gốc EMBER2018 có 1000 cây, đây là chuẩn
        standard_trees = 1000
        
        if num_trees1 < num_trees2:
            if num_trees1 < standard_trees:
                logger.warning(f"  ⚠️  Model 1 có ít cây hơn ({num_trees1:,} vs {num_trees2:,}) - Model 1 THIẾU cây (chuẩn: {standard_trees:,})")
            else:
                logger.info(f"  ✅ Model 1 có ít cây hơn nhưng vẫn đủ ({num_trees1:,} vs {num_trees2:,})")
        elif num_trees1 > num_trees2:
            if num_trees2 < standard_trees:
                logger.warning(f"  ⚠️  Model 2 có ít cây hơn ({num_trees2:,} vs {num_trees1:,}) - Model 2 THIẾU cây (chuẩn: {standard_trees:,})")
            else:
                logger.info(f"  ✅ Model 2 có ít cây hơn nhưng vẫn đủ ({num_trees2:,} vs {num_trees1:,})")
        else:
            if num_trees1 == standard_trees:
                logger.info(f"  ✅ Số cây bằng nhau và ĐỦ ({num_trees1:,} cây - chuẩn)")
            else:
                logger.warning(f"  ⚠️  Số cây bằng nhau nhưng THIẾU ({num_trees1:,} vs chuẩn: {standard_trees:,})")
        
        # Đánh giá kích thước
        if size1_mb < size2_mb * 0.8:
            logger.warning(f"  ⚠️  Model 1 nhỏ hơn nhiều ({size1_mb:.1f} MB vs {size2_mb:.1f} MB) - Có thể THIẾU thông tin")
        elif size1_mb > size2_mb * 1.2:
            logger.info(f"  ℹ️  Model 1 lớn hơn ({size1_mb:.1f} MB vs {size2_mb:.1f} MB) - Có thể chứa nhiều thông tin hơn")
        else:
            logger.info(f"  ✅ Kích thước tương đương ({size1_mb:.1f} MB vs {size2_mb:.1f} MB)")
        
        # Đánh giá features
        if num_features1 != num_features2:
            logger.warning(f"  ❌ Số features khác nhau! ({num_features1} vs {num_features2}) - Có thể không tương thích")
        else:
            logger.info(f"  ✅ Số features giống nhau ({num_features1}) - Tương thích")
        
        # Tổng kết
        logger.info("")
        logger.info("🎯 TỔNG KẾT:")
        if num_trees1 == standard_trees and num_trees2 < standard_trees:
            logger.info(f"  ✅ Model 1 ĐỦ ({num_trees1:,} cây) - Model 2 THIẾU ({num_trees2:,} cây)")
        elif num_trees2 == standard_trees and num_trees1 < standard_trees:
            logger.info(f"  ✅ Model 2 ĐỦ ({num_trees2:,} cây) - Model 1 THIẾU ({num_trees1:,} cây)")
        elif num_trees1 == standard_trees and num_trees2 == standard_trees:
            logger.info(f"  ✅ Cả 2 model ĐỦ ({standard_trees:,} cây)")
        else:
            logger.warning(f"  ⚠️  Cả 2 model đều THIẾU (Model 1: {num_trees1:,}, Model 2: {num_trees2:,}, Chuẩn: {standard_trees:,})")
    
    def compare_performance(self, data_dir=None, feature_version=2, sample_size=10000):
        """
        So sánh hiệu năng của 2 model trên test set
        
        Args:
            data_dir: Đường dẫn đến data/ember2018
            feature_version: Version của features
            sample_size: Số samples để test (None = tất cả, hoặc số để test nhanh)
        """
        logger.info("=" * 80)
        logger.info("SO SÁNH HIỆU NĂNG TRÊN TEST SET")
        logger.info("=" * 80)
        
        try:
            import ember
            import numpy as np
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix
            )
            
            # Xác định data directory
            if data_dir is None:
                data_dir = self.project_root / "data" / "ember2018"
            else:
                data_dir = Path(data_dir)
            
            if not data_dir.exists():
                logger.error(f"Thư mục dataset không tồn tại: {data_dir}")
                return None
            
            logger.info(f"Đang load test set từ: {data_dir}")
            
            # Load test set
            X_test, y_test = ember.read_vectorized_features(
                str(data_dir), subset="test", feature_version=feature_version
            )
            
            logger.info(f"Test set: {X_test.shape[0]:,} samples")
            
            # Lấy sample nếu cần
            if sample_size and sample_size < len(y_test):
                logger.info(f"Chỉ test với {sample_size:,} samples đầu tiên (để test nhanh)")
                indices = np.random.choice(len(y_test), sample_size, replace=False)
                X_test = X_test[indices]
                y_test = y_test[indices]
            
            # Dự đoán với model 1
            logger.info(f"Đang dự đoán với Model 1 ({self.model1_path.name})...")
            y_pred1 = self.model1.predict(X_test)
            y_pred1_binary = (y_pred1 > 0.5).astype(int)
            
            # Dự đoán với model 2
            logger.info(f"Đang dự đoán với Model 2 ({self.model2_path.name})...")
            y_pred2 = self.model2.predict(X_test)
            y_pred2_binary = (y_pred2 > 0.5).astype(int)
            
            # Tính metrics cho model 1
            acc1 = accuracy_score(y_test, y_pred1_binary)
            prec1 = precision_score(y_test, y_pred1_binary)
            rec1 = recall_score(y_test, y_pred1_binary)
            f1_1 = f1_score(y_test, y_pred1_binary)
            auc1 = roc_auc_score(y_test, y_pred1)
            cm1 = confusion_matrix(y_test, y_pred1_binary)
            
            # Tính metrics cho model 2
            acc2 = accuracy_score(y_test, y_pred2_binary)
            prec2 = precision_score(y_test, y_pred2_binary)
            rec2 = recall_score(y_test, y_pred2_binary)
            f1_2 = f1_score(y_test, y_pred2_binary)
            auc2 = roc_auc_score(y_test, y_pred2)
            cm2 = confusion_matrix(y_test, y_pred2_binary)
            
            # Hiển thị so sánh
            logger.info("")
            logger.info("=" * 80)
            logger.info("KẾT QUẢ SO SÁNH HIỆU NĂNG")
            logger.info("=" * 80)
            logger.info(f"{'Metric':<20} | {'Model 1':<15} | {'Model 2':<15} | {'Khác biệt':<15} | {'Tốt hơn'}")
            logger.info("-" * 80)
            
            metrics = [
                ('Accuracy', acc1, acc2),
                ('Precision', prec1, prec2),
                ('Recall', rec1, rec2),
                ('F1-Score', f1_1, f1_2),
                ('AUC', auc1, auc2),
            ]
            
            for name, val1, val2 in metrics:
                diff = val2 - val1
                better = "Model 2" if diff > 0 else "Model 1" if diff < 0 else "Bằng nhau"
                logger.info(f"{name:<20} | {val1:>15.4f} | {val2:>15.4f} | {diff:>+15.4f} | {better}")
            
            # So sánh Confusion Matrix
            logger.info("")
            logger.info("CONFUSION MATRIX:")
            logger.info(f"Model 1: TN={cm1[0,0]:,} FP={cm1[0,1]:,} FN={cm1[1,0]:,} TP={cm1[1,1]:,}")
            logger.info(f"Model 2: TN={cm2[0,0]:,} FP={cm2[0,1]:,} FN={cm2[1,0]:,} TP={cm2[1,1]:,}")
            
            # Đánh giá tổng thể
            logger.info("")
            logger.info("🎯 KẾT LUẬN:")
            if auc1 > auc2:
                logger.info(f"  ✅ Model 1 TỐT HƠN về AUC ({auc1:.4f} vs {auc2:.4f})")
            elif auc2 > auc1:
                logger.info(f"  ✅ Model 2 TỐT HƠN về AUC ({auc2:.4f} vs {auc1:.4f})")
            else:
                logger.info(f"  ✅ Hai model TƯƠNG ĐƯƠNG về AUC ({auc1:.4f})")
            
            if rec1 > rec2:
                logger.info(f"  ✅ Model 1 phát hiện nhiều Malware hơn (Recall: {rec1:.4f} vs {rec2:.4f})")
            elif rec2 > rec1:
                logger.info(f"  ✅ Model 2 phát hiện nhiều Malware hơn (Recall: {rec2:.4f} vs {rec1:.4f})")
            
            return {
                'model1': {
                    'accuracy': acc1, 'precision': prec1, 'recall': rec1,
                    'f1': f1_1, 'auc': auc1, 'confusion_matrix': cm1
                },
                'model2': {
                    'accuracy': acc2, 'precision': prec2, 'recall': rec2,
                    'f1': f1_2, 'auc': auc2, 'confusion_matrix': cm2
                }
            }
            
        except Exception as e:
            logger.error(f"Lỗi so sánh hiệu năng: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def compare_on_files(self, test_files_dir=None):
        """
        So sánh 2 model trên các file thực tế
        
        Args:
            test_files_dir: Thư mục chứa các file để test
        """
        if test_files_dir is None:
            test_files_dir = self.project_root / "test_files"
        else:
            test_files_dir = Path(test_files_dir)
        
        if not test_files_dir.exists():
            logger.warning(f"Thư mục test files không tồn tại: {test_files_dir}")
            return
        
        logger.info("=" * 80)
        logger.info("SO SÁNH TRÊN FILE THỰC TẾ")
        logger.info("=" * 80)
        
        try:
            import ember
            
            # Tìm các file PE
            test_files = list(test_files_dir.glob("*.exe")) + list(test_files_dir.glob("*.dll"))
            
            if not test_files:
                logger.warning("Không tìm thấy file .exe hoặc .dll để test")
                return
            
            logger.info(f"Tìm thấy {len(test_files)} file để test")
            logger.info("")
            logger.info(f"{'File':<40} | {'Model 1 Score':<15} | {'Model 2 Score':<15} | {'Khác biệt':<15}")
            logger.info("-" * 80)
            
            differences = []
            for file_path in test_files:
                try:
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    if file_data[:2] != b'MZ':
                        continue
                    
                    # Predict với cả 2 model
                    score1 = ember.predict_sample(self.model1, file_data, feature_version=2)
                    score2 = ember.predict_sample(self.model2, file_data, feature_version=2)
                    
                    diff = abs(score2 - score1)
                    differences.append(diff)
                    
                    pred1 = 'Malware' if score1 > 0.5 else 'Benign'
                    pred2 = 'Malware' if score2 > 0.5 else 'Benign'
                    
                    status = ""
                    if pred1 != pred2:
                        status = " ⚠️  KHÁC NHAU!"
                    
                    logger.info(f"{file_path.name:<40} | {score1:>15.4f} | {score2:>15.4f} | {diff:>15.4f}{status}")
                    
                except Exception as e:
                    logger.warning(f"Không thể test file {file_path.name}: {e}")
            
            if differences:
                avg_diff = sum(differences) / len(differences)
                logger.info("")
                logger.info(f"Độ khác biệt trung bình: {avg_diff:.4f}")
                if avg_diff > 0.1:
                    logger.warning("  ⚠️  Hai model cho kết quả khác nhau đáng kể!")
                else:
                    logger.info("  ✅ Hai model cho kết quả tương đương")
            
        except Exception as e:
            logger.error(f"Lỗi so sánh trên file: {e}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description='So sánh 2 model EMBER với nhau',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # So sánh 2 model
  python -m train.compare_models -m1 model1.txt -m2 model2.txt
  
  # So sánh và đánh giá hiệu năng
  python -m train.compare_models -m1 model1.txt -m2 model2.txt --evaluate
  
  # So sánh nhanh với 10k samples
  python -m train.compare_models -m1 model1.txt -m2 model2.txt --evaluate --sample-size 10000
  
  # So sánh trên file thực tế
  python -m train.compare_models -m1 model1.txt -m2 model2.txt --test-files test_files/
        """
    )
    
    parser.add_argument(
        '-m1', '--model1',
        type=str,
        required=True,
        help='Đường dẫn đến model 1'
    )
    
    parser.add_argument(
        '-m2', '--model2',
        type=str,
        required=True,
        help='Đường dẫn đến model 2'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='So sánh hiệu năng trên test set'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10000,
        help='Số samples để test khi đánh giá (mặc định: 10000, dùng None để test tất cả)'
    )
    
    parser.add_argument(
        '--test-files',
        type=str,
        help='Thư mục chứa file thực tế để test'
    )
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo comparator
        comparator = ModelComparator(args.model1, args.model2)
        
        # So sánh thông tin cơ bản
        comparator.compare_basic_info()
        
        # So sánh hiệu năng nếu yêu cầu
        if args.evaluate:
            comparator.compare_performance(sample_size=args.sample_size)
        
        # So sánh trên file thực tế nếu có
        if args.test_files:
            comparator.compare_on_files(args.test_files)
        else:
            # Thử tìm thư mục test_files mặc định
            comparator.compare_on_files()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("SO SÁNH HOÀN TẤT!")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"Lỗi: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nĐã dừng bởi người dùng")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

