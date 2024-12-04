import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

# Judul aplikasi
st.header('Animal Classification CNN Model')

# Daftar nama hewan sesuai prediksi model
animal_names = ['Lobster', 'Udang', 'kepiting']

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
        outcome = f"The image belongs to '{animal_names[np.argmax(result)]}' with a score of {np.max(result) * 100:.2f}%"
        return outcome

    # Komponen upload file pada Streamlit
    uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Buat direktori untuk menyimpan file jika belum ada
        upload_dir = 'upload'
        os.makedirs(upload_dir, exist_ok=True)

        # Simpan file yang di-upload
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Tampilkan gambar yang di-upload
        st.image(file_path, width=200, caption="Uploaded Image")

        # Tampilkan hasil klasifikasi
        st.markdown(classify_images(file_path))
