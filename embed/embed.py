
from sentence_transformers import SentenceTransformer

vi_bge = SentenceTransformer("AITeamVN/Vietnamese_Embedding")  # 1024-dim
# Khuyến nghị: chuẩn hóa để dùng cosine
def embed_texts(texts):
    return vi_bge.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

q_vec = embed_texts(["Đạo đức là gì?"])[0]


# ví dụ
text = input("Nhập câu: ")
vec = embed_texts([text])[0]
print(len(vec), vec[:10])