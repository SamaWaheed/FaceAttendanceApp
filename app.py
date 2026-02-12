import streamlit as st
import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

attendance_set = set()

if "attendance_df" not in st.session_state:
    st.session_state["attendance_df"] = pd.DataFrame(columns=["Name", "Date", "Time"])

st.title("🔥 Face Attendance System 🔥")
st.write("Camera ON → Recognize faces → Mark attendance automatically")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_img)
        encodings = face_recognition.face_encodings(rgb_img, faces)
        
        for encoding, face_loc in zip(encodings, faces):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            top, right, bottom, left = face_loc
            cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(img, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if name != "Unknown" and name not in attendance_set:
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file_name = f"Attendance_{timestamp}.csv"

st.download_button(
    label="Download Attendance CSV",
    data=csv,
    file_name=csv_file_name,
    mime="text/csv"
)
