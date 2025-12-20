#!/usr/bin/env python3
"""
Script để test EMBER model đã train
Hỗ trợ test một file hoặc nhiều file trong thư mục
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
        logging.FileHandler('ember_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmberTester:
    """Class để test EMBER model"""
    
    def __init__(self, model_path, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.model_path = Path(model_path) if not Path(model_path).is_absolute() else Path(model_path)
        
        # Nếu model_path là relative, tìm trong project_root
        if not self.model_path.is_absolute():
            self.model_path = self.project_root / self.model_path
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file không tồn tại: {self.model_path}")
        
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Project root: {self.project_root}")
        
        # Setup import path
        project_root_str = str(self.project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        
        # Load model
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load LightGBM model"""
        try:
            import lightgbm as lgb
            logger.info("Đang load model...")
            self.model = lgb.Booster(model_file=str(self.model_path))
            
            # Hiển thị thông tin model
            num_trees = self.model.num_trees()
            num_features = self.model.num_feature()
            logger.info(f"✓ Model đã load thành công")
            logger.info(f"  - Số cây: {num_trees:,}")
            logger.info(f"  - Số features: {num_features:,}")
            
        except Exception as e:
            logger.error(f"✗ Lỗi load model: {e}")
            raise
    
    def is_pe_file(self, file_path):
        """
        Kiểm tra xem file có phải PE file không
        
        Logic:
        - File phải có MZ header (bắt buộc)
        - Nếu có PE signature thì tốt (PE file hoàn chỉnh)
        - Nếu không có PE signature nhưng có MZ header thì vẫn cho phép (file PE không hoàn chỉnh hoặc file test)
        """
        try:
            file_path = Path(file_path)
            # Kiểm tra file size (file quá nhỏ không thể là PE hợp lệ)
            if file_path.stat().st_size < 2:
                return False
            
            with open(file_path, 'rb') as f:
                header = f.read(2)
                # PE file bắt đầu với 'MZ' (DOS header) - BẮT BUỘC
                if header != b'MZ':
                    return False
                
                # Kiểm tra PE signature nếu file đủ lớn (>= 64 bytes)
                file_size = file_path.stat().st_size
                if file_size >= 64:
                    try:
                        # Đọc offset PE signature (ở offset 0x3C)
                        f.seek(0x3C)
                        pe_offset_bytes = f.read(4)
                        if len(pe_offset_bytes) == 4:
                            pe_offset = int.from_bytes(pe_offset_bytes, byteorder='little')
                            # Kiểm tra offset hợp lệ
                            if 0 < pe_offset < file_size:
                                f.seek(pe_offset)
                                pe_signature = f.read(4)
                                # Nếu có PE signature thì chắc chắn là PE file
                                if pe_signature == b'PE\x00\x00':
                                    return True
                    except Exception:
                        # Nếu không đọc được PE signature, vẫn cho phép nếu có MZ header
                        pass
                
                # Nếu có MZ header thì vẫn cho phép (có thể là file PE không hoàn chỉnh hoặc file test)
                return True
        except Exception:
            return False
    
    def predict_file(self, file_path, feature_version=2):
        """Dự đoán một file PE"""
        try:
            import ember
            
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File không tồn tại: {file_path}")
                return None
            
            # Kiểm tra xem có phải file PE không
            is_valid_pe = self.is_pe_file(file_path)
            if not is_valid_pe:
                logger.warning(f"⚠️  File '{file_path.name}' không phải file PE hợp lệ!")
                logger.warning("EMBER chỉ phân tích file PE (Portable Executable): .exe, .dll, .sys, .scr, v.v.")
                logger.warning("File PE phải bắt đầu với 'MZ' header.")
                return None
            
            # Kiểm tra xem có PE signature đầy đủ không (cảnh báo nếu không có)
            try:
                with open(file_path, 'rb') as f:
                    if file_path.stat().st_size >= 64:
                        f.seek(0x3C)
                        pe_offset_bytes = f.read(4)
                        if len(pe_offset_bytes) == 4:
                            pe_offset = int.from_bytes(pe_offset_bytes, byteorder='little')
                            file_size = file_path.stat().st_size
                            if 0 < pe_offset < file_size:
                                f.seek(pe_offset)
                                pe_signature = f.read(4)
                                if pe_signature != b'PE\x00\x00':
                                    logger.warning(f"⚠️  File '{file_path.name}' có MZ header nhưng không có PE signature đầy đủ")
                                    logger.warning("File có thể là file PE không hoàn chỉnh hoặc file test. Vẫn sẽ thử phân tích...")
            except Exception:
                pass  # Bỏ qua nếu không kiểm tra được
            
            # Kiểm tra file size (file quá lớn có thể gây memory issue)
            file_size = file_path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100 MB
            if file_size > max_size:
                logger.warning(f"⚠️  File quá lớn ({file_size / (1024**2):.1f} MB > {max_size / (1024**2):.1f} MB)")
                logger.warning("File quá lớn có thể gây memory issue. Bỏ qua file này.")
                return None
            
            logger.info(f"Đang phân tích: {file_path.name} ({file_size / 1024:.1f} KB)")
            
            # Đọc file dưới dạng binary
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Dự đoán
            # Score từ LightGBM là xác suất malware (0.0 = chắc chắn Benign, 1.0 = chắc chắn Malware)
            # Threshold: score > 0.5 → Malware, score <= 0.5 → Benign
            score = ember.predict_sample(self.model, file_data, feature_version=feature_version)
            score = float(score)
            
            # Xác định prediction và confidence
            prediction = 'Malware' if score > 0.5 else 'Benign'
            confidence = max(score, 1 - score) * 100  # Confidence: phần trăm chắc chắn
            
            # Cảnh báo nếu score gần threshold (có thể nhầm lẫn)
            warning = None
            if 0.4 <= score <= 0.6:
                warning = "⚠️  Score gần ngưỡng (0.4-0.6) - Có thể cần kiểm tra thêm!"
            elif prediction == 'Benign' and score > 0.3:
                warning = "⚠️  Benign nhưng score khá cao (>0.3) - Có thể là suspicious file"
            elif prediction == 'Malware' and score < 0.7:
                warning = "⚠️  Malware nhưng score thấp (<0.7) - Có thể là false positive"
            
            return {
                'file': file_path.name,
                'path': str(file_path),
                'score': score,
                'prediction': prediction,
                'confidence': confidence,
                'warning': warning,
                'size': len(file_data)
            }
            
        except Exception as e:
            error_msg = str(e)
            if 'bad_format' in error_msg or 'AttributeError' in error_msg:
                logger.error(f"Lỗi LIEF version: {error_msg}")
                logger.error("Đây có thể do:")
                logger.error("  1. File không phải PE hợp lệ")
                logger.error("  2. LIEF version không tương thích (đã được sửa)")
                logger.error("  3. File bị corrupt hoặc không đọc được")
            else:
                logger.error(f"Lỗi khi phân tích {file_path}: {e}")
            return None
    
    def predict_directory(self, directory, feature_version=2, extensions=None):
        """Dự đoán nhiều file trong thư mục"""
        if extensions is None:
            extensions = ['.exe', '.dll', '.sys', '.scr', '.com', '.bat', '.cmd']
        
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Thư mục không tồn tại: {directory}")
            return []
        
        logger.info(f"Đang quét thư mục: {directory}")
        
        results = []
        files = list(directory.glob('*'))
        total_files = sum(1 for f in files if f.is_file() and f.suffix.lower() in extensions)
        
        logger.info(f"Tìm thấy {total_files} file PE để phân tích...")
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in extensions:
                continue
            
            result = self.predict_file(file_path, feature_version)
            if result:
                results.append(result)
                # Hiển thị kết quả ngay
                print(f"  {result['file']:40} | {result['prediction']:8} | Score: {result['score']:.4f}")
        
        return results
    
    def print_results(self, results, save_csv=False):
        """Hiển thị kết quả và lưu nếu cần"""
        if not results:
            logger.warning("Không có kết quả nào!")
            return
        
        logger.info("=" * 80)
        logger.info("KẾT QUẢ PHÂN TÍCH")
        logger.info("=" * 80)
        
        # Phân loại
        malware_count = sum(1 for r in results if r['prediction'] == 'Malware')
        benign_count = len(results) - malware_count
        
        logger.info(f"Tổng số file: {len(results)}")
        logger.info(f"  - Malware: {malware_count} ({malware_count/len(results)*100:.1f}%)")
        logger.info(f"  - Benign:  {benign_count} ({benign_count/len(results)*100:.1f}%)")
        logger.info("=" * 80)
        
        # Chi tiết từng file
        logger.info("\nChi tiết:")
        logger.info(f"{'File':<45} | {'Kết quả':<8} | {'Score':<8} | {'Confidence':<10} | {'Size (KB)':<10}")
        logger.info("-" * 100)
        
        for result in results:
            size_kb = result['size'] / 1024
            confidence = result.get('confidence', max(result['score'], 1 - result['score']) * 100)
            warning = result.get('warning', '')
            
            log_line = f"{result['file']:<45} | {result['prediction']:<8} | {result['score']:<8.4f} | {confidence:<10.1f}% | {size_kb:<10.2f}"
            if warning:
                log_line += f" {warning}"
            logger.info(log_line)
        
        # Top malware (score cao nhất)
        if malware_count > 0:
            logger.info("\n⚠️  TOP MALWARE (Score cao nhất):")
            malware_results = [r for r in results if r['prediction'] == 'Malware']
            malware_results.sort(key=lambda x: x['score'], reverse=True)
            for i, result in enumerate(malware_results[:10], 1):
                logger.info(f"  {i}. {result['file']:<45} Score: {result['score']:.4f}")
        
        # Lưu CSV nếu yêu cầu
        if save_csv:
            try:
                import pandas as pd
                df = pd.DataFrame(results)
                csv_path = self.project_root / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                logger.info(f"\n✓ Kết quả đã lưu: {csv_path}")
            except ImportError:
                logger.warning("Không có pandas, bỏ qua lưu CSV")
            except Exception as e:
                logger.error(f"Lỗi lưu CSV: {e}")
    
    def test_sample(self):
        """Test với file mẫu được tạo sẵn"""
        logger.info("Tạo file PE mẫu để test...")
        try:
            import ember
            
            # Tạo file PE mẫu (header cơ bản)
            pe_header = b'MZ' + b'\x00' * 58 + b'PE\x00\x00' + b'\x00' * 1000
            test_file = self.project_root / 'test_sample.exe'
            
            with open(test_file, 'wb') as f:
                f.write(pe_header)
            
            logger.info(f"Đã tạo file test: {test_file}")
            
            # Test
            result = self.predict_file(test_file)
            if result:
                self.print_results([result])
            
            # Xóa file test
            if test_file.exists():
                test_file.unlink()
                logger.info("Đã xóa file test")
            
        except Exception as e:
            logger.error(f"Lỗi test mẫu: {e}")
    
    def evaluate_model_quality(self, data_dir=None, feature_version=2, sample_size=None):
        """
        Đánh giá chất lượng model trên test set từ dataset EMBER2018
        
        Args:
            data_dir: Đường dẫn đến thư mục data/ember2018 (mặc định: project_root/data/ember2018)
            feature_version: Version của features (mặc định: 2)
            sample_size: Số samples để test (None = tất cả, hoặc số cụ thể để test nhanh)
        
        Returns:
            dict: Các metrics đánh giá
        """
        logger.info("=" * 80)
        logger.info("ĐÁNH GIÁ CHẤT LƯỢNG MODEL")
        logger.info("=" * 80)
        
        try:
            import ember
            import numpy as np
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix,
                classification_report
            )
            
            # Xác định data directory
            if data_dir is None:
                data_dir = self.project_root / "data" / "ember2018"
            else:
                data_dir = Path(data_dir)
            
            if not data_dir.exists():
                logger.error(f"Thư mục dataset không tồn tại: {data_dir}")
                logger.error("Cần có dataset EMBER2018 để đánh giá model")
                return None
            
            logger.info(f"Đang load test set từ: {data_dir}")
            
            # Load test set (memory-mapped)
            X_test, y_test = ember.read_vectorized_features(
                str(data_dir), subset="test", feature_version=feature_version
            )
            
            logger.info(f"Test set: {X_test.shape[0]:,} samples x {X_test.shape[1]:,} features")
            
            # Lấy sample nếu cần (để test nhanh)
            if sample_size and sample_size < len(y_test):
                logger.info(f"Chỉ test với {sample_size:,} samples đầu tiên (để test nhanh)")
                indices = np.random.choice(len(y_test), sample_size, replace=False)
                X_test = X_test[indices]
                y_test = y_test[indices]
            
            # Dự đoán trên test set
            logger.info("Đang dự đoán trên test set (có thể mất vài phút)...")
            y_pred = self.model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Tính các metrics
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary)
            recall = recall_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)
            auc = roc_auc_score(y_test, y_pred)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_binary)
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            
            # False Positive Rate và False Negative Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Hiển thị kết quả
            logger.info("=" * 80)
            logger.info("KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG MODEL")
            logger.info("=" * 80)
            logger.info(f"Test set size: {len(y_test):,} samples")
            logger.info("")
            logger.info("📊 METRICS CHÍNH:")
            logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"  Precision: {precision:.4f} ({precision*100:.2f}%) - Trong số dự đoán Malware, {precision*100:.2f}% đúng")
            logger.info(f"  Recall:    {recall:.4f} ({recall*100:.2f}%) - Phát hiện được {recall*100:.2f}% số Malware thực tế")
            logger.info(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%) - Cân bằng giữa Precision và Recall")
            logger.info(f"  AUC:       {auc:.4f} ({auc*100:.2f}%) - Khả năng phân biệt Malware/Benign")
            logger.info("")
            logger.info("📋 CONFUSION MATRIX:")
            logger.info(f"                    Dự đoán Benign    Dự đoán Malware")
            logger.info(f"  Thực tế Benign:   {tn:>10,} (TN)    {fp:>10,} (FP)")
            logger.info(f"  Thực tế Malware:  {fn:>10,} (FN)    {tp:>10,} (TP)")
            logger.info("")
            logger.info("⚠️  TỶ LỆ LỖI:")
            logger.info(f"  False Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%) - Báo sai Malware")
            logger.info(f"  False Negative Rate (FNR): {fnr:.4f} ({fnr*100:.2f}%) - Bỏ sót Malware")
            logger.info("")
            
            # Đánh giá chất lượng
            logger.info("🎯 ĐÁNH GIÁ CHẤT LƯỢNG:")
            if auc >= 0.99:
                logger.info("  ✅ AUC >= 0.99: XUẤT SẮC! Model rất tốt")
            elif auc >= 0.95:
                logger.info("  ✅ AUC >= 0.95: TỐT! Model có chất lượng cao")
            elif auc >= 0.90:
                logger.info("  ⚠️  AUC >= 0.90: KHÁ TỐT, nhưng có thể cải thiện")
            else:
                logger.info("  ❌ AUC < 0.90: CẦN CẢI THIỆN model")
            
            if precision >= 0.95:
                logger.info("  ✅ Precision >= 0.95: Ít false positive (báo sai Malware)")
            else:
                logger.info(f"  ⚠️  Precision < 0.95: Có {fp:,} false positives")
            
            if recall >= 0.90:
                logger.info("  ✅ Recall >= 0.90: Phát hiện tốt Malware")
            else:
                logger.info(f"  ⚠️  Recall < 0.90: Bỏ sót {fn:,} malware ({fnr*100:.2f}%)")
            
            logger.info("=" * 80)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'fpr': fpr,
                'fnr': fnr,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            }
            
        except Exception as e:
            logger.error(f"Lỗi đánh giá model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description='Test EMBER malware detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Test một file
  python -m train.test_ember_model -m ember_model_pycharm.txt -f sample.exe
  
  # Test cả thư mục
  python -m train.test_ember_model -m ember_model_pycharm.txt -d C:\\samples
  
  # Test và lưu kết quả CSV
  python -m train.test_ember_model -m ember_model_pycharm.txt -d C:\\samples --csv
  
  # Đánh giá chất lượng model trên test set (200k samples)
  python -m train.test_ember_model -m ember_model_pycharm.txt --evaluate
  
  # Đánh giá nhanh với 10k samples
  python -m train.test_ember_model -m ember_model_pycharm.txt --evaluate --sample-size 10000
        """
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='ember_model_pycharm.txt',
        help='Đường dẫn đến file model (mặc định: ember_model_pycharm.txt)'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Đường dẫn đến file PE cần test'
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        help='Đường dẫn đến thư mục chứa các file PE cần test'
    )
    
    parser.add_argument(
        '-v', '--feature-version',
        type=int,
        default=2,
        help='Feature version của EMBER (mặc định: 2)'
    )
    
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Lưu kết quả vào file CSV'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Test với file mẫu được tạo tự động'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Đánh giá chất lượng model trên test set từ dataset EMBER2018'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Số samples để test khi đánh giá (mặc định: tất cả, dùng số nhỏ để test nhanh)'
    )
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo tester
        tester = EmberTester(args.model)
        
        results = []
        
        # Test file đơn
        if args.file:
            result = tester.predict_file(args.file, args.feature_version)
            if result:
                results.append(result)
        
        # Test thư mục
        elif args.directory:
            results = tester.predict_directory(args.directory, args.feature_version)
        
        # Đánh giá chất lượng model
        elif args.evaluate:
            metrics = tester.evaluate_model_quality(sample_size=args.sample_size)
            if metrics:
                logger.info("✓ Đánh giá hoàn tất!")
            return
        
        # Test mẫu
        elif args.sample:
            tester.test_sample()
            return
        
        # Nếu không có tham số, test mẫu
        else:
            logger.info("Không có file hoặc thư mục được chỉ định, chạy test mẫu...")
            tester.test_sample()
            return
        
        # Hiển thị kết quả
        if results:
            tester.print_results(results, save_csv=args.csv)
        else:
            logger.warning("Không có kết quả nào!")
    
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
