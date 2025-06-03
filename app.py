import os
import re
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from googleapiclient.discovery import build

# =====================
# CẤU HÌNH API
# =====================
API_KEY = 'AIzaSyBce8zkHaX5SC42SM2tBKu_RZzrMY46UUo'  # ← Nhập YouTube API Key tại đây
youtube = build('youtube', 'v3', developerKey=API_KEY)

# =====================
# TẢI MÔ HÌNH TIẾNG VIỆT
# =====================
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ['tiêu cực', 'trung tính', 'tích cực']

def analyze_sentiment(comment):
    encoded_input = tokenizer(comment, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = F.softmax(output.logits, dim=1)
    return labels[torch.argmax(scores).item()]

def is_spam(comment):
    return bool(re.search(r"(http|www\.|zalo|telegram|free|subscribe|vay tiền|kiếm tiền)", comment.lower()))

def search_videos(keyword, max_results=3):
    request = youtube.search().list(q=keyword, part='id', type='video', maxResults=max_results)
    response = request.execute()
    return [item['id']['videoId'] for item in response['items']]

def get_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100, textFormat='plainText')
    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    return comments

def classify_all_comments(all_comments):
    results = []
    for comment in all_comments:
        label = "spam" if is_spam(comment) else analyze_sentiment(comment)
        results.append({"comment": comment, "label": label})
    return pd.DataFrame(results)

def save_to_excel(df):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Tất cả", index=False)
        for label in df["label"].unique():
            df[df["label"] == label].to_excel(writer, sheet_name=label.capitalize(), index=False)
    output.seek(0)
    return output

# =====================
# GIAO DIỆN WEB
# =====================
st.set_page_config(page_title="Phân tích bình luận YouTube", layout="centered")
st.title("💬 Phân tích bình luận YouTube (Tiếng Việt)")
st.markdown("Ứng dụng sử dụng mô hình học máy để phân loại cảm xúc của bình luận.")

keyword = st.text_input("🔍 Nhập từ khóa tìm kiếm video:")

if st.button("Phân tích"):
    if not keyword:
        st.warning("Vui lòng nhập từ khóa.")
    else:
        with st.spinner("Đang tìm video và tải bình luận..."):
            video_ids = search_videos(keyword)
            all_comments = []
            for vid in video_ids:
                comments = get_comments(vid)
                all_comments.extend(comments)

        if not all_comments:
            st.error("Không tìm thấy bình luận.")
        else:
            st.success(f"Đã tải {len(all_comments)} bình luận. Đang phân tích...")
            df = classify_all_comments(all_comments)

            st.dataframe(df.head(10))
            file = save_to_excel(df)

            st.download_button(
                label="📥 Tải file Excel",
                data=file,
                file_name="comments_vi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )