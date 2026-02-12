import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

attendance_set = set()

if "attendance_df" not in st.session_state:
    st.session_state["attendance_df"] = pd.DataFrame(columns=["Name", "Date", "Time"])

st.title("🔥 Face Attendance System 🔥")
st.write("Camera ON → Detect faces → Mark attendance automatically")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # تسجيل كل وجه مرة واحدة فقط
            name = "Person"  # ممكن تعملي ID لكل شخص بعدين لو حابة
            if name not in attendance_set:
                attendance_set.add(name)
                now = datetime.now()
                st.session_state["attendance_df"] = pd.concat([
                    st.session_state["attendance_df"],
                    pd.DataFrame([[name, now.date(), now.time().replace(microsecond=0)]], columns=["Name", "Date", "Time"])
                ], ignore_index=True)

        return img

webrtc_streamer(key="face-attendance", video_transformer_factory=VideoTransformer)

st.subheader("📋 Attendance Table")
st.dataframe(st.session_state["attendance_df"])

csv = st.session_state["attendance_df"].to_csv(index=False).encode('utf-8')
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
csv_file_name = f"Attendance_{timestamp}.csv"

st.download_button(
    label="Download Attendance CSV",
    data=csv,
    file_name=csv_file_name,
    mime="text/csv"
)
