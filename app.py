import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

# Judul aplikasi
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Animal Classification CNN Model</h1>",
    unsafe_allow_html=True
)

# Deskripsi aplikasi
st.markdown(
    """
    <p style='text-align: center; font-size: 18px; color: #555;'>
    Unggah gambar hewan laut (kepiting, lobster, atau udang) dan model kami akan mengklasifikasikannya!
    </p>
    """,
    unsafe_allow_html=True
)

# Daftar nama hewan sesuai prediksi model
animal_names = ['kepiting', 'lobster', 'udang']

# Periksa keberadaan file model sebelum memuat
model_path = 'animal_Recog_Model.h5'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' tidak ditemukan. Pastikan file tersebut ada di direktori yang benar.")
else:
    model = load_model(model_path)

    # Fungsi untuk mengklasifikasikan gambar
    def classify_images(image_path):
        # Memuat dan memproses gambar
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        # Melakukan prediksi menggunakan model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        outcome = f"The image belongs to '<b>{animal_names[np.argmax(result)]}</b>' with a score of {np.max(result) * 100:.2f}%"
        return outcome

    # Komponen upload file pada Streamlit
    st.markdown("<h3 style='color: #4CAF50;'>Upload Your Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Buat direktori untuk menyimpan file jika belum ada
        upload_dir = 'upload'
        os.makedirs(upload_dir, exist_ok=True)

        # Simpan file yang di-upload
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Tampilkan gambar yang di-upload
        st.image(file_path, use_column_width=True, caption="Uploaded Image")

        # Tampilkan hasil klasifikasi
        st.markdown(
            f"<div style='text-align: center; font-size: 20px; color: #333;'><b>Classification Result:</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align: center; font-size: 18px; color: #555;'>{classify_images(file_path)}</p>",
            unsafe_allow_html=True
        )
