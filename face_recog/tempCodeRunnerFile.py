import streamlit as st
import numpy as np
import face_recognition
import os
import cv2
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_var = (1.0 - face_distance) / (range_val * 2.0)
    
    if face_distance > face_match_threshold:
        return str(round(linear_var * 100, 2)) + '%'
    else:
        value = (linear_var + ((1.0 - linear_var) * (math.pow(linear_var - 0.5, 2) * 2 + 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    face_confidences = []

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        directory = r'E:\Face_recognition-main\face_recog\faces'  # Change this to your directory containing known faces
        for image in os.listdir(directory):
            full_path = os.path.join(directory, image)
            face_image = face_recognition.load_image_file(full_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

    def run_recognition(self, video_capture):
        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                self.face_confidences = []

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')
                    self.face_confidences.append(confidence)

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name, confidence in zip(self.face_locations, self.face_names, self.face_confidences):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                color = (0, 0, 255)

                if confidence.endswith('%'):
                    confidence_value = float(confidence[:-1])
                    if confidence_value >= 85:
                        color = (0, 255, 0)
                    elif confidence_value >= 75:
                        color = (0, 165, 255)
                    else:
                        color = (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the annotated frame using Streamlit
            st.image(frame, channels="RGB", use_column_width=True)

            # Check if the "Stop" button is clicked
            if st.button("Stop"):
                break

def main():
    fr = FaceRecognition()
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        st.error("Error: Unable to open video source.")
    else:
        fr.run_recognition(video_capture)

if __name__ == "__main__":
    main()
