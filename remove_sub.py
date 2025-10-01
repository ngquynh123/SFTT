import json

with open("data/qa600_sub.json", "r", encoding="utf-8") as f:
    data = json.load(f)

result = []
for block in data:
    for q in block.get("questions", []):
        result.append(q)

with open("data/qa600.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Đã chuyển đổi xong, tổng số câu hỏi: {len(result)}")