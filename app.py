import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

file_id = "1L_Btmvlo4YHIE6wV-WgOkWdndiD_ng4p"
url = f"https://drive.google.cgdowom/uc?id={file_id}"
       
       
model_path = "trained_potato_plant_disease_model-1.keras"

if not os.path.exists(model_path):
    st.warning("Downloading Model form Google Drive...")
    gdown.download(url,model_path,quiet=False)

st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())

def load_model():
    
    return tf.keras.models.load_model(model_path)

def model_prediction(test_image):
    try:
        model = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    confidence_scores = predictions[0] * 100
    return np.argmax(predictions),confidence_scores


class_name = ['Potato_Early_Blight', 'Potato_Late_Blight', 'Potato_Healthy']

disease_info = {
    "Potato_Early_Blight": "⚠️ Early blight is caused by Alternaria solani fungus. Symptoms include brown leaf spots. Use fungicides like chlorothalonil.",
    "Potato_Late_Blight": "⚠️ Late blight is caused by Phytophthora infestans. Leaves develop dark patches. Treat with copper-based fungicides.",
    "Potato_Healthy": "✅ Your plant is healthy! Keep monitoring for any symptoms."
}

st.set_page_config(page_title="Potato Leaf Disease Detection", page_icon="🌱", layout="wide")


st.sidebar.title("🌱 Potato Plant Disease Detection")
app_mode = st.sidebar.radio('Select Page', ['🏠 Home', '🔍 Disease Recognition'])



if app_mode == '🏠 Home':
    st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🌾 Potato Leaf Disease Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Using AI to detect diseases in potato leaves for sustainable farming. 🚜</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Developed for sustainable agriculture 🌱</p>", unsafe_allow_html=True)
    img=Image.open('Diseases.jpg')
    st.image(img,use_container_width=True)
    

elif app_mode == '🔍 Disease Recognition':
    st.markdown("<h2 style='text-align:center;'>🔍 Upload or Capture an Image</h2>", unsafe_allow_html=True)
    test_image = st.file_uploader("📤 Upload an Image:", type=['jpg', 'png', 'jpeg'])
    if st.button("📸 Take a Photo"):
        test_image = st.camera_input("Capture an Image")
    st.markdown(
    """
    <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,)


    if test_image:
        
        st.image(test_image, caption="📸 Uploaded Image", use_container_width=True)
        if st.button('🔎 Predict'):
            st.snow() 

        
            with st.spinner('🔍 Analyzing... Please wait...'):
                result_index, confidence_scores = model_prediction(test_image)
            
                st.success(f'✅ Prediction: {class_name[result_index]} ({confidence_scores[result_index]:.2f}%)')
                st.markdown("### Confidence Scores")
                st.bar_chart(dict(zip(class_name, confidence_scores)))
                st.info(disease_info[class_name[result_index]])
    else:
        st.warning("⚠️ Please upload an image to proceed!")

