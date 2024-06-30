import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque
import time

# Initialize Mediapipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Prepare the updated data
bounding_boxes = np.array([
    [135, 114],
    [200, 185],
    [113, 104],
    [185, 154],
    [360, 270],
    [280, 220],
    [164, 145],
    [245, 215],
    [430, 330],
    [135, 110],
    [104, 90],
    [95, 83],
    [88, 75],
    [78, 68],
    [193, 160],
    [297, 232]
])
distances = np.array([58, 36, 70, 45, 20, 28, 50, 30, 17, 60, 80, 90, 100, 110, 40, 25])

# Fit the linear regression model
model = LinearRegression()
model.fit(bounding_boxes, distances)

# Average face dimensions in mm
MALE_FACE_WIDTH = 147.6
FEMALE_FACE_WIDTH = 140.1

# Global variable to adjust lens size
LENS_SIZE_MULTIPLIER = 1.7


# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Function to estimate the distance of the face from the camera
def estimate_distance(height, width):
    return model.predict([[height, width]])[0]


# Function to draw dashed lines
def draw_dashed_line(img, start, end, color, thickness, dash_length=5):
    x1, y1 = start
    x2, y2 = end
    line_length = calculate_distance(start, end)
    num_dashes = int(line_length // dash_length)
    for i in range(num_dashes):
        start_x = int(x1 + (x2 - x1) * (i / num_dashes))
        start_y = int(y1 + (y2 - y1) * (i / num_dashes))
        end_x = int(x1 + (x2 - x1) * ((i + 1) / num_dashes))
        end_y = int(y1 + (y2 - y1) * ((i + 1) / num_dashes))
        if i % 2 == 0:
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)


# Set gender (can be detected or specified)
gender = 'male'  # or 'female'

# Determine known face width based on gender
if gender == 'male':
    KNOWN_FACE_WIDTH = MALE_FACE_WIDTH
elif gender == 'female':
    KNOWN_FACE_WIDTH = FEMALE_FACE_WIDTH
else:
    raise ValueError("Invalid gender specified. Please choose 'male' or 'female'.")

# Queue to store distance measurements
distance_queue = deque(maxlen=30)  # Assuming 6 FPS, this stores last 5 seconds of data

# Timer to control the update of the metrics
last_update_time = time.time()

# Variables to store metrics
eye_width = 0
eye_distance = 0
face_width = 0

# Start the webcam and process frames
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7) as face_mesh:
    while cap.isOpened():
        current_time = time.time()
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect face landmarks
        results = face_mesh.process(frame_rgb)

        # Draw face landmarks and calculate measurements
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Extract landmark points for calculations
                landmarks = face_landmarks.landmark

                # Calculate bounding box of the face
                h, w, _ = frame.shape
                x_min = int(min([landmark.x * w for landmark in landmarks]))
                x_max = int(max([landmark.x * w for landmark in landmarks]))
                y_min = int(min([landmark.y * h for landmark in landmarks]))
                y_max = int(max([landmark.y * h for landmark in landmarks]))
                face_width_in_frame = x_max - x_min
                face_height_in_frame = y_max - y_min

                # Enlarge the bounding box for display
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                enlarged_width = int(face_width_in_frame * 1.9)
                enlarged_height = int(face_height_in_frame * 1.9)
                x_min_enlarged = x_center - enlarged_width // 2
                x_max_enlarged = x_center + enlarged_width // 2
                y_min_enlarged = y_center - enlarged_height // 2
                y_max_enlarged = y_center + enlarged_height // 2

                # Estimate distance from camera
                distance = estimate_distance(face_height_in_frame, face_width_in_frame)
                distance_queue.append(distance)

                # Update metrics every 5 seconds
                if current_time - last_update_time >= 5:
                    # Calculate the average distance over the last 5 seconds
                    avg_distance = sum(distance_queue) / len(distance_queue)

                    # Adjust measurements based on distance
                    scale_factor = KNOWN_FACE_WIDTH / face_width_in_frame

                    # Calculate and print adjusted measurements
                    eye_width = calculate_distance((landmarks[33].x * w, landmarks[33].y * h),
                                                   (landmarks[133].x * w, landmarks[133].y * h)) * scale_factor / 10

                    eye_distance = calculate_distance((landmarks[133].x * w, landmarks[133].y * h),
                                                      (landmarks[362].x * w, landmarks[362].y * h)) * scale_factor / 10

                    face_width = calculate_distance((landmarks[454].x * w, landmarks[454].y * h),
                                                    (landmarks[234].x * w, landmarks[234].y * h)) * scale_factor / 10

                    last_update_time = current_time

                # Calculate eye centers
                left_eye_center = (
                    int((landmarks[33].x + landmarks[133].x) / 2 * w),
                    int((landmarks[33].y + landmarks[133].y) / 2 * h))
                right_eye_center = (
                    int((landmarks[362].x + landmarks[263].x) / 2 * w),
                    int((landmarks[362].y + landmarks[263].y) / 2 * h))

                # Get coordinates for the outer edge of the glasses
                left_glass_leg_end = (int(landmarks[234].x * w), int(landmarks[234].y * h))
                right_glass_leg_end = (int(landmarks[454].x * w), int(landmarks[454].y * h))

                # Offset starting points of the eye distance lines
                left_eye_start = (left_eye_center[0], left_eye_center[1] - 10)
                right_eye_start = (right_eye_center[0], right_eye_center[1] - 10)
                left_eye_top = (left_eye_center[0], left_eye_center[1] - 60)
                right_eye_top = (right_eye_center[0], right_eye_center[1] - 60)

                # Draw dashed lines for eye distance
                draw_dashed_line(frame, left_eye_start, left_eye_top, (0, 0, 0), 1)
                draw_dashed_line(frame, right_eye_start, right_eye_top, (0, 0, 0), 1)

                # Eye distance text shifted a bit more to the left
                eye_distance_text_pos = (
                (left_eye_top[0] + right_eye_top[0]) // 2 - 30, (left_eye_top[1] + right_eye_top[1]) // 2)
                cv2.putText(frame, f"{eye_distance:.2f} cm", eye_distance_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)

                # Calculate eye centers and sizes
                left_eye_width = calculate_distance((landmarks[133].x * w, landmarks[133].y * h),
                                                    (landmarks[33].x * w, landmarks[33].y * h))
                right_eye_width = calculate_distance((landmarks[362].x * w, landmarks[362].y * h),
                                                     (landmarks[263].x * w, landmarks[263].y * h))

                glass_radius = int(
                    max(left_eye_width, right_eye_width) / 2 * LENS_SIZE_MULTIPLIER)  # Adjustable lens size

                # Draw glasses
                # Left lens
                cv2.circle(frame, left_eye_center, glass_radius, (255, 0, 0), 2)
                # Right lens
                cv2.circle(frame, right_eye_center, glass_radius, (255, 0, 0), 2)
                # Bridge
                cv2.line(frame, (left_eye_center[0] + glass_radius, left_eye_center[1]),
                         (right_eye_center[0] - glass_radius, right_eye_center[1]), (255, 0, 0), 2)
                # Left arm
                cv2.line(frame, (left_eye_center[0] - glass_radius, left_eye_center[1]),
                         left_glass_leg_end, (255, 0, 0), 2)
                # Right arm
                cv2.line(frame, (right_eye_center[0] + glass_radius, right_eye_center[1]),
                         right_glass_leg_end, (255, 0, 0), 2)

                # Draw dashed lines from the left and right sides of the inside circle of the left eye, reduced by 45%
                line_reduction_factor = 0.55
                left_eye_inside_left = (int(landmarks[33].x * w), int(landmarks[33].y * h))
                left_eye_inside_right = (int(landmarks[133].x * w), int(landmarks[133].y * h))
                left_eye_below_left = (
                left_eye_inside_left[0], int(left_eye_inside_left[1] + 60 * line_reduction_factor))
                left_eye_below_right = (
                left_eye_inside_right[0], int(left_eye_inside_right[1] + 60 * line_reduction_factor))
                draw_dashed_line(frame, left_eye_inside_left, left_eye_below_left, (0, 0, 0), 1)
                draw_dashed_line(frame, left_eye_inside_right, left_eye_below_right, (0, 0, 0), 1)

                # Eye width text shifted down and to the left
                eye_width_text_pos = ((left_eye_below_left[0] + left_eye_below_right[0]) // 2 - 30,
                                      (left_eye_below_left[1] + left_eye_below_right[1]) // 2 + 20)
                cv2.putText(frame, f"{eye_width:.2f} cm", eye_width_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                            2)

                # Draw enlarged bounding box
                cv2.rectangle(frame, (x_min_enlarged, y_min_enlarged), (x_max_enlarged, y_max_enlarged), (0, 0, 255), 2)

                # Draw dashed lines for face width
                face_left = (int(landmarks[234].x * w), int(landmarks[234].y * h))
                face_right = (int(landmarks[454].x * w), int(landmarks[454].y * h))
                face_below_left = (face_left[0], int(face_left[1] + 140 * line_reduction_factor))
                face_below_right = (face_right[0], int(face_right[1] + 140 * line_reduction_factor))
                draw_dashed_line(frame, face_left, face_below_left, (0, 0, 0), 1)
                draw_dashed_line(frame, face_right, face_below_right, (0, 0, 0), 1)

                # Face width text moved a tiny bit to the left and down
                face_width_text_pos = ((face_below_left[0] + face_below_right[0]) // 2 - 40,
                                       (face_below_left[1] + face_below_right[1]) // 2 + 20)
                cv2.putText(frame, f"{face_width:.2f} cm", face_width_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)

                # Display the live estimated distance from the camera
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                            2)

        # Display the frame with landmarks, glasses, and bounding box
        cv2.imshow('Face Landmarks with Glasses', frame)

        # Press 'q' to quit the webcam window
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
