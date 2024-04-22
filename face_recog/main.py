import numpy as np
import face_recognition
import os,sys
import cv2
import math
from PIL import Image
import streamlit as st

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
        directory = r'E:\Face_recognition-main\face_recog\faces'
        for image in os.listdir(directory):
            full_path = os.path.join(directory, image)
            face_image = face_recognition.load_image_file(full_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        # Create a video capture object for the default camera (index 0)
        video_capture = cv2.VideoCapture(0)

        # Check if the video capture object was initialized successfully
        if not video_capture.isOpened():
            # Print error message and exit
            sys.exit("Video source not found...")

        while True:
            # Read a single frame from the video stream
            ret, frame = video_capture.read()

            # Check if processing of the current frame is required
            if self.process_current_frame:
                # Resize the frame
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the frame to RGB format
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all faces in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                self.face_confidences = []
            for face_encoding in self.face_encodings:
                # Compare the current face encoding with known face encodings
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

                # Default name and confidence level for the current face
                name = "Unknown"
                confidence = "Unknown"

                # Calculate the face distances from known faces
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                # Find the index of the best match (the smallest distance)
                best_match_index = np.argmin(face_distances)

                # Check if there's a match with any known face
                if matches[best_match_index]:
                    # Retrieve the name and confidence level associated with the best match
                    name = self.known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

                # Append the identified name and confidence level to the face names list
                self.face_names.append(f'{name} ({confidence})')
                self.face_confidences.append(confidence)

            self.process_current_frame = not self.process_current_frame
            for (top, right, bottom, left), name, confidence in zip(self.face_locations, self.face_names, self.face_confidences):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                color = (0, 0, 255)  # Red color by default

                # Modify the color based on the confidence level
                if confidence.endswith('%'):
                    confidence_value = float(confidence[:-1])
                    if confidence_value >= 85:
                        color = (0, 255, 0)  # Green color
                    elif confidence_value >= 75:
                        color = (0, 165, 255)  # Orange color
                    else:
                        color = (0, 0, 255)  # Red color

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw a filled rectangle as the background for the face name
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, -1)

                # Write the face name on the frame
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Display the annotated frame
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

# Streamlit app

def main():
    st.title('Face Recognition App')

    # Run face recognition
    fr = FaceRecognition()

    # Display the video stream with face recognition
    st.write('Face Recognition Live Feed')
    stframe = st.empty()
    
    start_button = st.button('Start Webcam')
    stop_button = st.button('Stop Webcam')

    # Initialize video capture object outside the loop
    video_capture = None

    while True:
        # Check if start button is pressed and video capture is not initiated
        if start_button and video_capture is None:
            video_capture = cv2.VideoCapture(0)
            start_button = False
            print("Webcam started")

        # Check if stop button is pressed and video capture is initiated
        if stop_button and video_capture is not None:
            video_capture.release()
            video_capture = None
            print("Webcam released")
            break

        # If video capture is initiated, read frames and run recognition
        if video_capture is not None:
            ret, frame = video_capture.read()
            fr.process_current_frame = True
            fr.run_recognition()

            # Display the annotated frame
            stframe.image(frame, channels="BGR")


if __name__ == "__main__":
    main()
