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

# 初始化回饋狀態記憶（避免網頁重新整理時按鈕狀態錯亂）
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# =====================================================================
# 2. 載入你上傳的 AI 模型大腦
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

# 如果使用者上傳了新圖片，重置回饋按鈕狀態
if uploaded_file is not None:
    # 這裡用圖片名稱當作檢查，如果換新圖片就讓按鈕可以重新點擊
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.feedback_submitted = False

    image = Image.open(uploaded_file)
    st.image(image, caption='📷 你上傳的待測圖片', use_container_width=True)
    
    with st.spinner("🔄 AI 正在深度分析畫風紋理、色彩與線條中..."):
        img = image.resize((224, 224))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0) 
        
        prediction = model.predict(img_array)[0]
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
    st.subheader(f"✨ 鑑定結論：這張圖片極有可能是【{labels[max_idx]}】！")
    
    # =====================================================================
    # 5. 新增：使用者正確與錯誤回饋按鈕
    # =====================================================================
    st.write("")
    st.write("### 📢 您覺得 AI 判斷得準確嗎？")
    st.write("您的回饋能幫助我們未來將模型優化得更好！")
    
    # 建立左右並排兩個按鈕
    col1, col2 = st.columns(2)
    
    with col1:
        correct_btn = st.button("👍 準確！完全正確", use_container_width=True, disabled=st.session_state.feedback_submitted)
    with col2:
        incorrect_btn = st.button("👎 猜錯了！不夠準確", use_container_width=True, disabled=st.session_state.feedback_submitted)
        
    # 按鈕點擊後的邏輯處裡
    if correct_btn:
        st.session_state.feedback_submitted = True
        st.balloons() # 正確的話噴出氣球慶祝！
        st.success("🎉 太棒了！感謝您的正面肯定，AI 家族感到非常驕傲！")
        
    if incorrect_btn:
        st.session_state.feedback_submitted = True
        st.info("💡 收到您的回饋！我們會將此圖片加入「困難樣本集」，留給 AI 下次好好罰寫重新學習。")
