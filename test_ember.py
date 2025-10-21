#!/usr/bin/env python3
"""
EMBER Test Script
Kiểm tra hoạt động của EMBER và demo các chức năng chính
"""

import os
import sys
import tempfile
import numpy as np

def test_ember_installation():
    """Kiểm tra cài đặt EMBER"""
    print("🔍 Kiểm tra cài đặt EMBER...")
    try:
        import ember
        print("✅ EMBER đã được cài đặt thành công!")
        return True
    except ImportError as e:
        print(f"❌ Lỗi cài đặt EMBER: {e}")
        return False

def test_lief_installation():
    """Kiểm tra cài đặt LIEF"""
    print("🔍 Kiểm tra cài đặt LIEF...")
    try:
        import lief
        print(f"✅ LIEF version: {lief.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Lỗi cài đặt LIEF: {e}")
        return False

def test_lightgbm_installation():
    """Kiểm tra cài đặt LightGBM"""
    print("🔍 Kiểm tra cài đặt LightGBM...")
    try:
        import lightgbm as lgb
        print("✅ LightGBM đã được cài đặt!")
        return True
    except ImportError as e:
        print(f"❌ Lỗi cài đặt LightGBM: {e}")
        return False

def create_sample_pe_file():
    """Tạo file PE mẫu để test"""
    print("📁 Tạo file PE mẫu...")
    
    # Tạo một file PE đơn giản (chỉ để test, không thực thi được)
    sample_pe = b'MZ' + b'\x00' * 58 + b'PE\x00\x00' + b'\x00' * 1000
    
    # Lưu vào file tạm
    temp_file = tempfile.NamedTemporaryFile(suffix='.exe', delete=False)
    temp_file.write(sample_pe)
    temp_file.close()
    
    print(f"✅ File mẫu đã tạo: {temp_file.name}")
    return temp_file.name

def test_feature_extraction():
    """Test trích xuất features"""
    print("🔧 Test trích xuất features...")
    try:
        import ember
        
        # Tạo file PE mẫu
        sample_file = create_sample_pe_file()
        
        # Đọc file
        with open(sample_file, 'rb') as f:
            file_data = f.read()
        
        # Tạo feature extractor
        extractor = ember.PEFeatureExtractor(feature_version=2)
        
        # Trích xuất features
        features = extractor.feature_vector(file_data)
        
        print(f"✅ Đã trích xuất {len(features)} features")
        print(f"📊 Một số features đầu tiên: {features[:10]}")
        
        # Dọn dẹp
        os.unlink(sample_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất features: {e}")
        return False

def test_raw_features():
    """Test trích xuất raw features"""
    print("🔧 Test trích xuất raw features...")
    try:
        import ember
        
        # Tạo file PE mẫu
        sample_file = create_sample_pe_file()
        
        # Đọc file
        with open(sample_file, 'rb') as f:
            file_data = f.read()
        
        # Tạo feature extractor
        extractor = ember.PEFeatureExtractor(feature_version=2)
        
        # Trích xuất raw features
        raw_features = extractor.raw_features(file_data)
        
        print("✅ Raw features đã được trích xuất:")
        for key, value in raw_features.items():
            if key != 'sha256':  # Bỏ qua SHA256 hash
                print(f"  - {key}: {type(value).__name__}")
        
        # Dọn dẹp
        os.unlink(sample_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất raw features: {e}")
        return False

def test_ember_functions():
    """Test các function chính của EMBER"""
    print("🔧 Test các function chính...")
    try:
        import ember
        
        # Test PEFeatureExtractor
        extractor = ember.PEFeatureExtractor(feature_version=2)
        print(f"✅ PEFeatureExtractor dim: {extractor.dim}")
        
        # Test các function khác
        print("✅ Các function EMBER có sẵn:")
        functions = [name for name in dir(ember) if not name.startswith('_')]
        for func in functions[:10]:  # Hiển thị 10 function đầu tiên
            print(f"  - {func}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi test functions: {e}")
        return False

def main():
    """Hàm chính"""
    print("=" * 60)
    print("🛡️  EMBER MALWARE DETECTION - TEST SCRIPT")
    print("=" * 60)
    print()
    
    # Danh sách các test
    tests = [
        ("Cài đặt EMBER", test_ember_installation),
        ("Cài đặt LIEF", test_lief_installation),
        ("Cài đặt LightGBM", test_lightgbm_installation),
        ("Functions EMBER", test_ember_functions),
        ("Trích xuất features", test_feature_extraction),
        ("Raw features", test_raw_features),
    ]
    
    # Chạy các test
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        print()
    
    # Kết quả tổng kết
    print("=" * 60)
    print("📊 KẾT QUẢ TỔNG KẾT")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 Tất cả test đều PASSED! EMBER hoạt động tốt!")
        print("\n📚 Bước tiếp theo:")
        print("1. Xem file HUONG_DAN_CHAY_DU_AN.md để biết cách sử dụng")
        print("2. Chạy: python ember_demo.py your_file.exe")
        print("3. Sử dụng Docker: docker run -it ember-malware-detection")
    else:
        print(f"\n⚠️  Có {total - passed} test FAILED!")
        print("\n🔧 Khắc phục:")
        print("1. Cài đặt lại dependencies: pip install -r requirements.txt")
        print("2. Sử dụng Docker: docker build -t ember-malware-detection .")
        print("3. Kiểm tra Python version: python --version")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
