import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# Load YOLO model
model = YOLO('yolov11.pt')  # Ganti dengan path model YOLO Anda

# Fungsi untuk melakukan prediksi
def predict_and_annotate(image):
    # Convert frame ke format OpenCV
    image_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Jalankan prediksi
    results = model.predict(image_np, conf=0.473)  # Threshold confidence
    
    # Anotasi hasil pada gambar, hanya menampilkan nama kelas (tanpa skor)
    annotated_image = results[0].plot(labels=True, conf=False)  # Menonaktifkan skor koefisien
    
    # Menyimpan kelas pertama yang terdeteksi
    if len(results[0].boxes.cls) > 0:
        predicted_class = results[0].names[results[0].boxes.cls[0].item()]  # Mengambil kelas pertama jika ada
    else:
        predicted_class = "Tidak Ada Daun yang Terdeteksi"

    # Convert kembali ke RGB untuk ditampilkan di Streamlit
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), predicted_class

# Fungsi untuk mengunduh gambar
def download_image(image, filename):
    img_bytes = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.download_button(
        label="Unduh Hasil Prediksi",
        data=img_bytes,
        file_name=filename,
        mime="image/png"
    )

# Tema desain alami dengan warna hijau dan elemen alam
st.set_page_config(page_title="Tomato Leaf Disease Predict", page_icon="🌿", layout="centered")

# Gaya untuk header dan teks
st.markdown(
    """
    <style>
    .main {
        background-color: #e0f7e9;  # Warna latar belakang hijau muda
    }
    .css-ffhzg2 {
        color: #004d00;
        font-family: 'Arial', sans-serif;
    }
    .css-145k2gt {
        color: #006400;
        font-family: 'Georgia', serif;
        font-size: 28px;
    }
    .stButton>button {
        background-color: #32CD32;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #228B22;
    }
    .stDownloadButton>button {
        background-color: #32CD32;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: 0.3s;
    }
    .stDownloadButton>button:hover {
        background-color: #006400;
    }
    .stTextInput>div>div>input {
        background-color: #d4f1d4;
    }
    .stFileUploader>div>div>input {
        background-color: #d4f1d4;
    }
    .stImage img {
        max-width: 100% !important;
        height: auto !important;
    }
    .green-box {
        background-color: #00712D;
        border: 2px solid #4b9c1f;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .green-box h1 {
        color: white;
        font-size: 30px;
        margin: 0;
    }
    .green-box p {
        color: white;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# Judul aplikasi dalam kotak hijau dengan teks di tengah
st.markdown('<div class="green-box"><h1>🌿 Prediksi Kesehatan Daun Tomat 🍅</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="green-box"><p>"Unggah gambar atau gunakan kamera untuk memulai deteksi kesehatan daun tanaman tomat secara langsung. Sistem ini akan memberikan informasi terkait kondisi kesehatan tanaman tomat Anda."</p></div>', unsafe_allow_html=True)

# Tab untuk pilihan input (upload gambar atau akses kamera)
tab1, tab2 = st.tabs(["Unggah Gambar", "Deteksi Langsung"])

with tab1:
    uploaded_file = st.file_uploader("Unggah gambar daun tomat:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Baca file gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        
        # Prediksi dan tampilkan hasil
        if st.button("Lakukan Prediksi", key="predict_button_upload"):
            with st.spinner("Memproses..."):
                result_image, predicted_class = predict_and_annotate(np.array(image))
            st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
            st.write(f"Hasil Deteksi: {predicted_class}")
            
            # Nama file berdasarkan kelas prediksi dan waktu
            filename = f"{predicted_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            download_image(result_image, filename)

with tab2:
    # Pilihan untuk akses kamera
    st.write("Menggunakan kamera untuk deteksi secara langsung.")
    start_camera = st.checkbox("Mulai Kamera/Ambil Ulang Gambar", key="start_camera_checkbox")
    
    # Tempat untuk menampilkan frame kamera
    FRAME_WINDOW = st.image([])

    # Akses kamera
    camera = cv2.VideoCapture(0)  # Akses kamera

    # Variabel untuk menyimpan status prediksi real-time
    capture_requested = False
    captured_image = None

    if start_camera:
        # Tangkap gambar jika tombol "Capture" ditekan
        if st.button("Capture", key="capture_button_outside"):
            capture_requested = True

        while start_camera:
            ret, frame = camera.read()
            if not ret:
                st.warning("Tidak dapat mengakses kamera.")
                break

            # Ubah format frame ke RGB dan lakukan prediksi
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, predicted_class = predict_and_annotate(frame_rgb)

            # Tampilkan frame dengan bounding box
            FRAME_WINDOW.image(processed_frame, use_container_width=True)

            # Tangkap gambar jika tombol "Capture" ditekan
            if capture_requested:
                captured_image = frame_rgb  # Simpan frame yang sedang ditampilkan
                capture_requested = False  # Reset status capture
                break

        # Setelah checkbox dinonaktifkan, lepaskan kamera
        if not start_camera:
            camera.release()

    # Tampilkan hasil gambar yang di-capture dan prediksi
    if captured_image is not None:
        st.write("Gambar yang diambil:")
        st.image(captured_image, caption="Gambar yang diambil", use_container_width=True)
        
        # Prediksi ulang pada gambar yang di-capture
        with st.spinner("Memproses gambar yang diambil..."):
            result_image, predicted_class = predict_and_annotate(captured_image)
        st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
        st.write(f"Hasil Deteksi: {predicted_class}")

        # Nama file berdasarkan kelas prediksi dan waktu
        filename = f"{predicted_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        download_image(result_image, filename)