import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np

# Thiết lập đường dẫn tới model đã train
model_path = 'runs/train/exp6/weights/best.pt'  # Thay đổi theo đường dẫn của bạn
path_load = 'ultralytics/yolov5'
# Tải model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

st.title("DETECT NINE DASH-LINE")

conf_thres = st.slider("Chọn ngưỡng độ tin cậy (Confidence Threshold)", 0.25, 0.5, 0.75)
model.conf = conf_thres

# Tải ảnh từ tệp
uploaded_file = st.file_uploader("Select Image or Video", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    # Xử lý ảnh hoặc video
    if uploaded_file.type in ['image/jpeg', 'image/png']:
        image = Image.open(uploaded_file)
        st.image(image, caption='File Uploaded', use_column_width=True)
        detected = False
        # Chuyển đổi ảnh thành định dạng phù hợp với YOLOv5
        img = np.array(image)
        results = model(img) 
        detected_classes = results.xyxyn[0][:, -1].numpy()
        class_names = results.names

        if any(class_names[int(cls)] == '9_dash_line' for cls in detected_classes):
            detected = True

        # Hiển thị kết quả
        st.image(results.render()[0], caption='Detect', use_column_width=True)
        if detected:
            st.write("**Discovered nine dash-line in this image!**")
        else:
            st.write("**Nothing**")

    elif uploaded_file.type == 'video/mp4':
        st.video(uploaded_file)

        # Lưu video vào tệp tạm thời
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        # Xử lý video
        vid = cv2.VideoCapture("temp_video.mp4")
        frame_count = 0
        detected = False
        first_detection_time = None  # Thời gian phát hiện đầu tiên

        fps = vid.get(cv2.CAP_PROP_FPS)  # Lấy thông số FPS của video
        count_print = False
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            results = model(frame) 
            detected_classes = results.xyxyn[0][:, -1].numpy()
            class_names = results.names

            if any(class_names[int(cls)] == '9_dash_line' for cls in detected_classes):
                if not detected:  # Chỉ ghi thời gian nếu chưa phát hiện trước đó
                    first_detection_time = frame_count / fps
                    detected = True
                # Hiển thị kết quả
                if not count_print:
                    count_print = True
                    st.image(results.render()[0], caption='Detect', use_column_width=True)

            frame_count += 1

        vid.release()
        
        if detected:
            st.write("**Discovered nine dash-line in this video!**")
            st.write(f"**First detection at: {first_detection_time:.2f} seconds**")
        else:
            st.write("**Nothing**")

        vid.release()