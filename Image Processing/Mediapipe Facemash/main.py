#pip install mediapipe opencv-python

import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh and Drawing
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize FaceMesh with options
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Detect only 1 face
        refine_landmarks=True,  # Better precision for landmarks around eyes, lips, etc.
        min_detection_confidence=0.5,  # Minimum confidence for detection
        min_tracking_confidence=0.5  # Minimum confidence for tracking
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image color from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect face mesh
        results = face_mesh.process(image_rgb)

        # Draw the face mesh landmarks on the image if any faces are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # Use TESSELATION or CONTOURS
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # Display the annotated image
        cv2.imshow('MediaPipe FaceMesh', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
