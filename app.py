import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = tf.keras.models.load_model('final_project/model/skin_disease_model_mobilenet.h5')

# Load class labels
try:
    class_labels = np.load('final_project/model/sd_test.npy', allow_pickle=True).item()
    if not isinstance(class_labels, dict):
        st.error("File 'sd_test.npy' tidak berisi dictionary. Periksa format file.")
        st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat class labels: {e}")
    st.stop()

# Membalik dictionary untuk memetakan indeks ke nama kelas
class_names = {v: k for k, v in class_labels.items()}

# Judul aplikasi
st.title("Aplikasi Deteksi Penyakit Kulit")
st.write("Unggah foto kulit Anda untuk mendeteksi penyakit kulit yang Anda alami.")

# Fungsi untuk memproses gambar
def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.convert('RGB')  # Pastikan format RGB
    img = img.resize((240, 240))  # Resize ke ukuran input model
    img_array = image.img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array

# File uploader
uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    st.write("")
    
    # Preproses gambar
    img_array = preprocess_image(uploaded_file)
    
    # Prediksi
    prediction = model.predict(img_array)

    # Ambil indeks kelas dengan probabilitas tertinggi
    predicted_class_index = int(np.argmax(prediction))  # Pastikan hasilnya integer
    predicted_class_name = class_names.get(predicted_class_index, "Unknown")  # Ambil nama kelas
    
    # Menampilkan hasil prediksi
    st.write("### Hasil Prediksi:")
    st.write(f"Class yang diprediksi: **{predicted_class_name}**")
    
    # Pesan khusus berdasarkan prediksi
    if predicted_class_name.lower() == "melanoma":
        st.write("Gambar ini terindikasi mengidap **Melanoma**, silakan konsultasi dengan dokter.")
    else:
        st.write(f"Pasien terindikasi mengalami **{predicted_class_name}**. Mohon konsultasi lebih lanjut.")
