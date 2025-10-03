#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo ra 100 câu hỏi ngẫu nhiên từ bên ngoài
Không liên quan đến môn học, chủ đề đa dạng
"""

import json
import random
import os
from pathlib import Path

# Ngân hàng câu hỏi tổng quát theo chủ đề
QUESTION_BANK = {
    "cuoc_song_hang_ngay": [
        "Bạn thường làm gì vào cuối tuần?",
        "Món ăn yêu thích của bạn là gì?",
        "Bạn có thích du lịch không?",
        "Thời tiết hôm nay như thế nào?",
        "Bạn thường dậy lúc mấy giờ?",
        "Sở thích của bạn là gì?",
        "Bạn có nuôi thú cưng không?",
        "Loại nhạc nào bạn thích nghe?",
        "Bạn có thích xem phim không?",
        "Màu sắc yêu thích của bạn là gì?",
        "Bạn thích mùa nào trong năm?",
        "Công việc mơ ước của bạn là gì?",
        "Bạn có thích nấu ăn không?",
        "Thành phố nào bạn muốn đến nhất?",
        "Bạn thường đọc sách không?"
    ],
    
    "cong_nghe": [
        "Smartphone đầu tiên bạn dùng là gì?",
        "Bạn thích iOS hay Android?",
        "AI sẽ thay đổi cuộc sống như thế nào?",
        "Mạng xã hội nào bạn dùng nhiều nhất?",
        "Bạn có sợ robot không?",
        "Internet đã thay đổi thế giới như thế nào?",
        "Game yêu thích của bạn là gì?",
        "Bạn có tin vào xe tự lái không?",
        "Công nghệ VR có tương lai không?",
        "Cryptocurrency có phải là tương lai?",
        "Bạn thích mua sắm online hay offline?",
        "Laptop hay desktop tốt hơn?",
        "5G có thực sự cần thiết không?",
        "Bạn có lo về bảo mật dữ liệu không?",
        "Smartwatch có hữu ích không?"
    ],
    
    "van_hoa_xa_hoi": [
        "Truyền thống nào của Việt Nam bạn yêu thích?",
        "Lễ hội nào bạn thích nhất?",
        "Gia đình có quan trọng không?",
        "Bạn nghĩ gì về văn hóa phương Tây?",
        "Ngôn ngữ nào khó học nhất?",
        "Bạn có thích tìm hiểu văn hóa nước khác không?",
        "Nghệ thuật có cần thiết trong đời sống?",
        "Âm nhạc truyền thống có còn giá trị?",
        "Bạn thích kiến trúc cổ hay hiện đại?",
        "Thế hệ trẻ có khác thế hệ trước?",
        "Tôn giáo có vai trò gì trong xã hội?",
        "Bạn nghĩ gì về hôn nhân đồng giới?",
        "Phụ nữ và nam giới có bình đẳng chưa?",
        "Giáo dục có đang thay đổi?",
        "Bạn có tin vào số phận không?"
    ],
    
    "thien_nhien": [
        "Động vật nào bạn thích nhất?",
        "Bạn có lo về biến đổi khí hậu không?",
        "Loài hoa nào đẹp nhất?",
        "Đại dương hay núi rừng hấp dẫn hơn?",
        "Bạn có thích cắm trại không?",
        "Thiên nhiên có cần được bảo vệ?",
        "Mùa mưa hay nắng dễ chịu hơn?",
        "Bạn có sợ động vật hoang dã không?",
        "Rừng Amazon có quan trọng không?",
        "Bạn thích ngắm sao không?",
        "Núi lửa có nguy hiểm không?",
        "Biển có bị ô nhiễm nghiêm trọng?",
        "Bạn có muốn sống gần thiên nhiên?",
        "Thức ăn hữu cơ có tốt hơn?",
        "Năng lượng tái tạo có khả thi không?"
    ],
    
    "giai_tri": [
        "Thể loại phim nào bạn thích?",
        "Ca sĩ nào bạn nghe nhiều nhất?",
        "Bạn có thích đi karaoke không?",
        "Sách hay phim thú vị hơn?",
        "Bạn thích comedy hay drama?",
        "Nghệ sĩ nào bạn thần tượng?",
        "Bạn có thích đi concert không?",
        "Game online hay offline hay hơn?",
        "Bạn thích xem thể thao nào?",
        "Nhạc Việt hay nhạc nước ngoài?",
        "Bạn có biết chơi nhạc cụ không?",
        "Phim hoạt hình có chỉ dành cho trẻ em?",
        "Reality show có thực tế không?",
        "Bạn thích đọc truyện tranh không?",
        "TikTok có ảnh hưởng gì đến giới trẻ?"
    ],
    
    "suc_khoe": [
        "Bạn có tập thể dục thường xuyên không?",
        "Yoga có tốt cho sức khỏe?",
        "Bạn ngủ mấy tiếng mỗi ngày?",
        "Stress có ảnh hưởng đến sức khỏe?",
        "Bạn có ăn chay không?",
        "Nước có quan trọng như thế nào?",
        "Bạn có hay bị đau đầu không?",
        "Vitamin có cần thiết không?",
        "Bạn thích tập gym hay chạy bộ?",
        "Sức khỏe tinh thần có quan trọng?",
        "Bạn có hay thức khuya không?",
        "Fast food có hại như người ta nói?",
        "Bạn có đi khám sức khỏe định kỳ?",
        "Thiền có giúp giảm stress?",
        "Bạn có thích massage không?"
    ],
    
    "du_lich": [
        "Quốc gia nào bạn muốn đến nhất?",
        "Bạn thích du lịch một mình hay theo nhóm?",
        "Khách sạn hay homestay tốt hơn?",
        "Bạn có sợ đi máy bay không?",
        "Du lịch trong nước hay nước ngoài?",
        "Bạn thích mang về quà gì khi du lịch?",
        "Túi xách hay vali tiện hơn?",
        "Bạn có thích chụp ảnh du lịch không?",
        "Mùa nào thích hợp du lịch nhất?",
        "Bạn có thích đi phượt không?",
        "Du lịch bụi có an toàn không?",
        "Bạn thích biển hay núi?",
        "Guide tour có cần thiết không?",
        "Bạn có muốn sống ở nước ngoài?",
        "Du lịch có tốn kém không?"
    ]
}

def generate_random_questions(n_questions=100):
    """Tạo ra n_questions câu hỏi ngẫu nhiên từ ngân hàng câu hỏi"""
    
    # Gộp tất cả câu hỏi từ các chủ đề
    all_questions = []
    for category, questions in QUESTION_BANK.items():
        for question in questions:
            all_questions.append({
                'question': question,
                'category': category
            })
    
    # Trộn ngẫu nhiên
    random.shuffle(all_questions)
    
    # Chọn số lượng câu hỏi yêu cầu
    if len(all_questions) <= n_questions:
        selected = all_questions
    else:
        selected = random.sample(all_questions, n_questions)
    
    return selected

def save_questions_json(questions, output_path):
    """Lưu câu hỏi ra file JSON"""
    
    # Thống kê theo category
    category_stats = {}
    for item in questions:
        cat = item['category']
        category_stats[cat] = category_stats.get(cat, 0) + 1
    
    output_data = {
        "description": "100 câu hỏi ngẫu nhiên đa chủ đề",
        "total_questions": len(questions),
        "categories": category_stats,
        "questions": [
            {
                "id": i+1, 
                "question": item['question'],
                "category": item['category']
            } 
            for i, item in enumerate(questions)
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã lưu {len(questions)} câu hỏi (JSON) tại: {output_path}")

def save_questions_txt(questions, output_path):
    """Lưu câu hỏi ra file text thuần"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("100 CÂU HỎI NGẪU NHIÊN ĐA CHỦ ĐỀ\n")
        f.write("=" * 50 + "\n\n")
        
        current_category = None
        for i, item in enumerate(questions, 1):
            # Hiển thị category nếu thay đổi
            if item['category'] != current_category:
                current_category = item['category']
                f.write(f"\n--- {current_category.upper().replace('_', ' ')} ---\n")
            
            f.write(f"{i:3d}. {item['question']}\n")
        
        f.write(f"\n\n{'='*50}\n")
        f.write("THỐNG KÊ THEO CHỦ ĐỀ:\n")
        
        # Thống kê
        category_stats = {}
        for item in questions:
            cat = item['category']
            category_stats[cat] = category_stats.get(cat, 0) + 1
        
        for cat, count in category_stats.items():
            percentage = (count / len(questions)) * 100
            f.write(f"- {cat.replace('_', ' ').title()}: {count} câu ({percentage:.1f}%)\n")
    
    print(f"✅ Đã lưu {len(questions)} câu hỏi (TXT) tại: {output_path}")

def main():
    # Setup
    output_dir = Path(r"D:\AI.LLM-khanh-no_rrf\data\test_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_questions = 100
    
    print("🎲 TẠO CÂU HỎI NGẪU NHIÊN ĐA CHỦ ĐỀ")
    print("=" * 60)
    print(f"🎯 Số câu hỏi: {n_questions}")
    print(f"📂 Thư mục output: {output_dir}")
    print("=" * 60)
    
    # Set random seed
    random.seed(42)
    
    # Thống kê ngân hàng câu hỏi
    total_available = sum(len(questions) for questions in QUESTION_BANK.values())
    print(f"\n📊 NGÂN HÀNG CÂU HỎI:")
    print(f"📝 Tổng số câu hỏi có sẵn: {total_available}")
    
    for category, questions in QUESTION_BANK.items():
        print(f"   - {category.replace('_', ' ').title()}: {len(questions)} câu")
    
    # Tạo câu hỏi ngẫu nhiên
    print(f"\n🎲 Tạo {n_questions} câu hỏi ngẫu nhiên...")
    selected_questions = generate_random_questions(n_questions)
    
    # Thống kê kết quả
    category_stats = {}
    for item in selected_questions:
        cat = item['category']
        category_stats[cat] = category_stats.get(cat, 0) + 1
    
    print(f"\n📊 PHÂN BỐ THEO CHỦ ĐỀ:")
    for category, count in sorted(category_stats.items()):
        percentage = (count / len(selected_questions)) * 100
        print(f"   - {category.replace('_', ' ').title()}: {count} câu ({percentage:.1f}%)")
    
    # Lưu file
    json_output = output_dir / "random_100_general_questions.json"
    txt_output = output_dir / "random_100_general_questions.txt"
    
    save_questions_json(selected_questions, json_output)
    save_questions_txt(selected_questions, txt_output)
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH!")
    print(f"📊 Đã tạo {len(selected_questions)} câu hỏi đa chủ đề")
    print(f"📄 File JSON: {json_output.name}")
    print(f"📄 File TXT: {txt_output.name}")
    print("=" * 60)
    
    # Preview
    print(f"\n🔍 PREVIEW 10 CÂU HỎI ĐẦU:")
    for i, item in enumerate(selected_questions[:10], 1):
        category_display = item['category'].replace('_', ' ').title()
        print(f"{i:2d}. [{category_display}] {item['question']}")

if __name__ == "__main__":
    main()