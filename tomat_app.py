import streamlit as st
import cv2
from PIL import Image
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov11.pt')  # Ganti dengan path model YOLO Anda

# Fungsi untuk melakukan prediksi dan anotasi gambar
def predict_and_annotate(image):
    image_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model.predict(image_np, conf=0.473)  # Threshold confidence
    annotated_image = results[0].plot(labels=True, conf=False)
    
    if len(results[0].boxes.cls) > 0:
        predicted_class = results[0].names[results[0].boxes.cls[0].item()]
    else:
        predicted_class = "Tidak Ada Daun yang Terdeteksi"

    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), predicted_class

# Fungsi untuk mengunduh gambar hasil prediksi
def download_image(image, filename):
    img_bytes = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.download_button(
        label="Unduh Hasil Prediksi",
        data=img_bytes,
        file_name=filename,
        mime="image/png"
    )

# Desain dan gaya aplikasi Streamlit
st.set_page_config(page_title="Prediksi Kesehatan Daun Tomat", page_icon="🌿", layout="centered")

st.markdown('<h1 style="text-align: center;">🌿 Prediksi Kesehatan Daun Tomat 🍅</h1>', unsafe_allow_html=True)

# Tab untuk memilih input
tab1, tab2 = st.tabs(["Unggah Gambar", "Deteksi Langsung"])

with tab1:
    uploaded_file = st.file_uploader("Unggah gambar daun tomat:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        
        if st.button("Lakukan Prediksi", key="predict_button_upload"):
            with st.spinner("Memproses..."):
                result_image, predicted_class = predict_and_annotate(np.array(image))
            st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
            st.write(f"Hasil Deteksi: {predicted_class}")
            filename = f"{predicted_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            download_image(result_image, filename)

with tab2:
    st.write("Menggunakan kamera untuk deteksi secara langsung.")
    
    # Checkbox untuk mengaktifkan atau menonaktifkan kamera
    start_camera = st.checkbox("Aktifkan Kamera", value=False)

    if start_camera:
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)  # Akses kamera
        capture_requested = False
        captured_image = None

        # Menangkap gambar jika tombol "Capture" ditekan
        if st.button("Capture", key="capture_button_outside"):
            capture_requested = True

        while start_camera:
            ret, frame = camera.read()
            if not ret:
                st.warning("Tidak dapat mengakses kamera.")
                break

            # Proses gambar dan lakukan prediksi
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, predicted_class = predict_and_annotate(frame_rgb)
            FRAME_WINDOW.image(processed_frame, use_container_width=True)

            if capture_requested:
                captured_image = frame_rgb
                capture_requested = False
                break

        # Setelah centang diuncheck, lepaskan kamera
        if not start_camera:
            camera.release()

        # Menampilkan gambar yang diambil dan hasil prediksi
        if captured_image is not None:
            st.write("Gambar yang diambil:")
            st.image(captured_image, caption="Gambar yang diambil", use_container_width=True)
            with st.spinner("Memproses gambar yang diambil..."):
                result_image, predicted_class = predict_and_annotate(captured_image)
            st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
            st.write(f"Hasil Deteksi: {predicted_class}")
            filename = f"{predicted_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            download_image(result_image, filename)
