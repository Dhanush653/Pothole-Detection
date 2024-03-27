import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load a model
model = YOLO("best600.pt")
class_names = model.names #roadpothole

# Function to draw bounding boxes and labels on the image
def draw_boxes(img, results):
    h, w, _ = img.shape
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img
# Function to process the video and display results
def process_video(video_file):
    # Save the uploaded video file
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())
    cap = cv2.VideoCapture("temp.mp4") # Open the saved video file

    if not cap.isOpened(): # Check if the video file was opened successfully
        st.error("Error opening video file.")
        return

    while True:
        #ret - boolean
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500)) # Resize frame

        results = model.predict(frame) # Perform YOLO prediction


        annotated_frame = draw_boxes(frame.copy(), results) # Draw bounding boxes and labels

        st.image(annotated_frame, channels="BGR", use_column_width=True)    # Display annotated frame

        if cv2.waitKey(1) != -1: # Check if any key event occurred
            break

    # Release video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Remove the temporary video file
    os.remove("temp.mp4")

    # Stop Streamlit script execution
    st.stop()

def main():
    # Add glowing text
    st.markdown(
        """
        <style>
        .glowing-text {
            font-family: 'Arial Black', sans-serif;
            font-size: 48px;
            text-align: center;
            animation: glowing 2s infinite;
        }
        @keyframes glowing {
            0% { color: #FF9933; } /* Saffron color */
            25% { color: #FFFFFF; } /* White color */
            50% { color: #128807; } /* Green color */
        }
        </style>
        <p class="glowing-text">Pothole Detection</p>
        """,
        unsafe_allow_html=True
    )

    # Add title and description
    st.write("Detect potholes in images or videos.")

    # Add button to choose detection mode
    detection_mode = st.radio("Select Detection Mode", ("Image", "Video"))

    if detection_mode == "Image":
        st.write("Upload an image:")
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            # Perform detection on the uploaded image
            image = np.array(Image.open(image_file))
            results = model.predict(image)
            annotated_image = draw_boxes(image.copy(), results)
            st.image(annotated_image, channels="BGR", use_column_width=True)

    elif detection_mode == "Video":
        st.write("Upload a video:")
        video_file = st.file_uploader("Upload Video", type=["mp4"])
        if video_file is not None:
            # Perform detection on the uploaded video
            process_video(video_file)

if __name__ == "__main__":
    main()
