#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chọn ngẫu nhiên 50 câu hỏi-đáp từ mỗi môn học
Đầu vào: D:\AI.LLM-khanh-no_rrf\data\dialogue\
Đầu ra: Random 50 QA pairs cho mỗi file JSON
"""

import json
import random
import os
from pathlib import Path

def load_json_file(filepath):
    """Load và parse file JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Lỗi đọc file {filepath}: {e}")
        return None

def extract_qa_pairs(data):
    """Trích xuất các cặp Q&A từ data JSON"""
    qa_pairs = []
    
    if isinstance(data, list):
        for item in data:
            qa = extract_single_qa(item)
            if qa:
                qa_pairs.append(qa)
    elif isinstance(data, dict):
        qa = extract_single_qa(data)
        if qa:
            qa_pairs.append(qa)
    
    return qa_pairs

def extract_single_qa(item):
    """Trích xuất Q&A từ một item"""
    qa = {}
    
    # Thử các key khác nhau cho question
    for q_key in ['question', 'q', 'query', 'instruction', 'prompt']:
        if q_key in item and item[q_key]:
            qa['question'] = str(item[q_key]).strip()
            break
    
    # Thử các key khác nhau cho answer
    for a_key in ['answer', 'a', 'response', 'output', 'reply']:
        if a_key in item and item[a_key]:
            qa['answer'] = str(item[a_key]).strip()
            break
    
    # Kiểm tra có đủ Q&A không
    if 'question' in qa and 'answer' in qa and qa['question'] and qa['answer']:
        return qa
    
    return None

def select_random_samples(qa_pairs, n_samples=50):
    """Chọn ngẫu nhiên n_samples từ danh sách QA"""
    if len(qa_pairs) <= n_samples:
        print(f"⚠️  Chỉ có {len(qa_pairs)} cặp QA, lấy tất cả")
        return qa_pairs
    
    return random.sample(qa_pairs, n_samples)

def save_selected_qa(qa_pairs, output_path, subject_name):
    """Lưu QA đã chọn ra file"""
    output_data = {
        "subject": subject_name,
        "total_selected": len(qa_pairs),
        "qa_pairs": qa_pairs
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã lưu {len(qa_pairs)} cặp QA cho môn {subject_name} tại: {output_path}")

def main():
    # Đường dẫn input và output
    input_dir = Path(r"D:\AI.LLM-khanh-no_rrf\data\dialogue")
    output_dir = Path(r"D:\AI.LLM-khanh-no_rrf\data\test_data\selected_qa")
    
    # Tạo thư mục output nếu chưa có
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Số câu cần chọn cho mỗi môn
    n_samples = 100
    
    print("🔍 BẮT ĐẦU CHỌN RANDOM QA")
    print("=" * 60)
    print(f"📂 Thư mục input: {input_dir}")
    print(f"📂 Thư mục output: {output_dir}")
    print(f"🎯 Số câu/môn: {n_samples}")
    print("=" * 60)
    
    # Set random seed để có thể reproduce
    random.seed(42)
    
    # Xử lý từng file JSON trong thư mục dialogue
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print("❌ Không tìm thấy file JSON nào trong thư mục!")
        return
    
    total_processed = 0
    
    for json_file in json_files:
        subject_name = json_file.stem  # Tên file không có extension
        print(f"\n🔄 Xử lý môn: {subject_name}")
        print(f"📄 File: {json_file.name}")
        
        # Load data
        data = load_json_file(json_file)
        if data is None:
            continue
        
        # Trích xuất QA pairs
        qa_pairs = extract_qa_pairs(data)
        print(f"📊 Tìm thấy {len(qa_pairs)} cặp QA")
        
        if not qa_pairs:
            print("⚠️  Không có QA pairs hợp lệ!")
            continue
        
        # Chọn random samples
        selected_qa = select_random_samples(qa_pairs, n_samples)
        
        # Lưu kết quả
        output_file = output_dir / f"{subject_name}_random_{len(selected_qa)}.json"
        save_selected_qa(selected_qa, output_file, subject_name)
        
        total_processed += 1
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH!")
    print(f"📊 Đã xử lý {total_processed} môn học")
    print(f"📂 Kết quả lưu tại: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()