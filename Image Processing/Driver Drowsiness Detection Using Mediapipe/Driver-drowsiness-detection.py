#pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# Initialize MediaPipe FaceMesh and Drawing
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Create custom drawing specs for smaller landmarks and connectors
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))  # Adjust thickness and radius

# Landmark indices for the left and right eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_EAR(eye_landmarks, landmarks):
    # Vertical distances
    vertical_1 = dist.euclidean(landmarks[eye_landmarks[1]], landmarks[eye_landmarks[5]])
    vertical_2 = dist.euclidean(landmarks[eye_landmarks[2]], landmarks[eye_landmarks[4]])

    # Horizontal distance
    horizontal = dist.euclidean(landmarks[eye_landmarks[0]], landmarks[eye_landmarks[3]])

    # Eye aspect ratio (EAR)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize FaceMesh
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Detect only one face
        refine_landmarks=True,  # High precision landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the frame to detect face mesh
        results = face_mesh.process(image_rgb)

        # If landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get landmark points for both eyes
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

                # Convert normalized landmarks to pixel coordinates
                h, w, _ = image.shape
                landmarks = np.array([(int(x * w), int(y * h)) for (x, y) in landmarks])

                # Calculate EAR for left and right eye
                left_EAR = calculate_EAR(LEFT_EYE, landmarks)
                right_EAR = calculate_EAR(RIGHT_EYE, landmarks)

                # Average EAR
                avg_EAR = (left_EAR + right_EAR) / 2.0

                # Draw the face landmarks with smaller points
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

                # Threshold for detecting closed eyes (typically EAR < 0.25)
                if avg_EAR < 0.25:
                    cv2.putText(image, "Eyes Closed", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(image, "Eyes Open", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Eye Closure Detection', image)

        # Break loop on pressing 'Esc'
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
