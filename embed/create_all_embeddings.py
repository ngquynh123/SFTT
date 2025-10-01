#!/usr/bin/env python3
"""
Script tạo embedding cho tất cả dữ liệu nhanh
"""
import json
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def create_embeddings():
    print("🔄 Loading embedding model...")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
    
    def embed_texts(texts):
        return model.encode(
            texts, 
            normalize_embeddings=True,
            batch_size=16,  # Giảm batch size để tránh memory issues
            show_progress_bar=True
        ).astype(np.float32)
    
    # Process all files in embed_data
    embed_data_dir = Path("../embed_data")
    
    for channel_dir in embed_data_dir.iterdir():
        if not channel_dir.is_dir():
            continue
            
        print(f"\n📁 Processing channel: {channel_dir.name}")
        
        for json_file in channel_dir.glob("*.json"):
            print(f"  🔄 Processing: {json_file.name}")
            
            try:
                # Load data
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    print(f"    ❌ File format not supported")
                    continue
                
                # Extract text for embedding
                texts = []
                for item in data[:100]:  # Limit to first 100 items for speed
                    text = ""
                    if "question" in item and "answer" in item:
                        text = f"Q: {item.get('question', '')} A: {item.get('answer', '')}"
                    elif "text" in item:
                        text = item.get("text", "")
                    else:
                        text = str(item)
                    
                    texts.append(text[:500])  # Limit text length
                
                if not texts:
                    print(f"    ❌ No text found")
                    continue
                
                print(f"    🔄 Creating embeddings for {len(texts)} items...")
                embeddings = embed_texts(texts)
                
                # Add embeddings to data
                for i, item in enumerate(data[:len(embeddings)]):
                    item["embedding"] = embeddings[i].tolist()
                
                # Save back
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data[:len(embeddings)], f, ensure_ascii=False, indent=2)
                
                print(f"    ✅ Saved {len(embeddings)} embeddings")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    print("\n🎉 Embedding creation completed!")

if __name__ == "__main__":
    create_embeddings()