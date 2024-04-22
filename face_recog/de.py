import streamlit as st
import cv2

def main():
    st.title("Webcam Streaming App")

    # Create buttons for starting and stopping the webcam
    start_button = st.button("Start")
    stop_button = st.button("Stop")

    # Placeholder for the video stream
    video_placeholder = st.empty()

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    # Variable to track whether webcam is streaming or not
    streaming = False

    # Main app logic
    while True:
        if start_button:
            streaming = True
        elif stop_button:
            streaming = False

        if streaming:
            ret, frame = cap.read()

            # Display the frame in Streamlit
            video_placeholder.image(frame, channels="BGR")

        # Check if the "Stop" button is clicked
        if stop_button:
            break

    # Release the webcam and close the app
    cap.release()

if __name__ == "__main__":
    main()
