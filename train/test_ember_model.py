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
        """Kiểm tra xem file có phải PE file không"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(2)
                # PE file bắt đầu với 'MZ'
                return header == b'MZ'
        except:
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
            if not self.is_pe_file(file_path):
                logger.warning(f"⚠️  File '{file_path.name}' không phải file PE hợp lệ!")
                logger.warning("EMBER chỉ phân tích file PE (Portable Executable): .exe, .dll, .sys, .scr, v.v.")
                logger.warning("File PE phải bắt đầu với 'MZ' header.")
                return None
            
            logger.info(f"Đang phân tích: {file_path.name}")
            
            # Đọc file dưới dạng binary
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Dự đoán
            score = ember.predict_sample(self.model, file_data, feature_version=feature_version)
            
            return {
                'file': file_path.name,
                'path': str(file_path),
                'score': float(score),
                'prediction': 'Malware' if score > 0.5 else 'Benign',
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
        logger.info(f"{'File':<50} | {'Kết quả':<8} | {'Score':<8} | {'Size (KB)':<10}")
        logger.info("-" * 80)
        
        for result in results:
            size_kb = result['size'] / 1024
            logger.info(f"{result['file']:<50} | {result['prediction']:<8} | {result['score']:<8.4f} | {size_kb:<10.2f}")
        
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


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description='Test EMBER malware detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Test một file
  python test_ember_model.py -m ember_model_pycharm.txt -f sample.exe
  
  # Test cả thư mục
  python test_ember_model.py -m ember_model_pycharm.txt -d C:\\samples
  
  # Test và lưu kết quả CSV
  python test_ember_model.py -m ember_model_pycharm.txt -d C:\\samples --csv
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
