import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =====================================================================
# 1. 網頁基本設定與標題
# =====================================================================
st.set_page_config(page_title="動漫風格鑑定器", page_icon="🎨", layout="centered")

st.title("🎨 AI 動漫插畫風格與畫風鑑定器")
st.write("上傳一張動漫劇照或插畫，讓 AI 幫你分析它是屬於吉卜力、美式卡通還是迪士尼風格！")

# =====================================================================
# 2. 載入你上傳的 AI 模型大腦 (直接讀取同資料夾底下的檔案)
# =====================================================================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('style_model.h5')

try:
    model = load_my_model()
    st.success("🤖 AI 風格辨識模組載入成功！")
except Exception as e:
    st.error("❌ 找不到 style_model.h5 模型檔案，請確保它與此程式碼放在同一個 GitHub 資料夾內。")

# =====================================================================
# 3. 網頁前端互動介面（上傳圖片）
# =====================================================================
uploaded_file = st.file_uploader("👉 請選擇或拖曳一張圖片上傳...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 你上傳的待測圖片', use_container_width=True)
    
    with st.spinner("🔄 AI 正在深度分析畫風紋理、色彩與線條中..."):
        # 將上傳的圖片調整為 224x224 大小
        img = image.resize((224, 224))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img) / 255.0  # 正規化
        img_array = np.expand_dims(img_array, axis=0) # 增加 Batch 維度
        
        # 進行預測
        prediction = model.predict(img_array)[0]
        
        # 標籤順序（依據資料夾字母排序：american, disney, ghibli）
        labels = ["美式卡通風格 (American)", "迪士尼風格 (Disney)", "吉卜力風格 (Ghibli)"]
        
    # =====================================================================
    # 4. 顯示分析結果
    # =====================================================================
    st.subheader("📊 畫風分析結果：")
    
    for label, prob in zip(labels, prediction):
        st.write(f"**{label}**")
        st.progress(float(prob))
        st.write(f"信心指數：{prob * 100:.2f}%")
        st.write("---")
        
    max_idx = np.argmax(prediction)
    st.balloons() # 噴出噴出慶祝氣球
    st.subheader(f"✨ 鑑定結論：這張圖片極有可能是【{labels[max_idx]}】！")
