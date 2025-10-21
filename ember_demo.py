#!/usr/bin/env python3
"""
EMBER Malware Detection Demo
Chạy trong Docker container để phân tích file PE
"""

import ember
import sys
import os

def main():
    print("=" * 50)
    print("    EMBER Malware Detection Demo")
    print("=" * 50)
    print()
    
    # Kiểm tra xem có file PE để phân tích không
    if len(sys.argv) < 2:
        print("Cách sử dụng: python ember_demo.py <path_to_pe_file>")
        print("Ví dụ: python ember_demo.py /workspace/sample.exe")
        return
    
    pe_file = sys.argv[1]
    
    if not os.path.exists(pe_file):
        print(f"ERROR: File không tồn tại: {pe_file}")
        return
    
    try:
        print(f"Đang phân tích file: {pe_file}")
        print("...")
        
        # Đọc file PE
        with open(pe_file, "rb") as f:
            file_data = f.read()
        
        # Tạo feature extractor
        extractor = ember.PEFeatureExtractor(2)  # Version 2 features
        
        # Trích xuất features
        features = extractor.feature_vector(file_data)
        
        print(f"✓ Đã trích xuất {len(features)} features")
        print(f"✓ Kích thước file: {len(file_data)} bytes")
        
        # Hiển thị một số features quan trọng
        print("\nMột số features quan trọng:")
        print(f"- General features: {features[:10]}")
        print(f"- Header features: {features[10:20]}")
        print(f"- Section features: {features[20:30]}")
        
        print("\n✓ Phân tích hoàn tất!")
        
    except Exception as e:
        print(f"ERROR: Lỗi khi phân tích file: {e}")

if __name__ == "__main__":
    main()
