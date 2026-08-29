

import cv2
import mediapipe as mp
import numpy as np
import os


IMAGE_PATHS = {
    "THUMBS_UP": "images/thumbs_up.jpg",
    "THINKING": "images/thinking.jpg",
    "SMART": "images/smart.jpg",
    "NEUTRAL": "images/neutral.jpg",
    "POINTING": "images/pointing.jpg",
    "PEACE": "images/peace.jpg",
}


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def load_and_resize_image(path, target_height):
    """
    Loads an image from disk and resizes it to match the camera frame height.
    Uses np.fromfile + cv2.imdecode instead of cv2.imread directly, since
    that combination handles tricky paths/encodings more reliably.
    """
    full_path = os.path.join(os.getcwd(), path)

    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return None

    img_array = np.fromfile(full_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Could not decode image: {path}")
        return None

    ratio = target_height / img.shape[0]
    target_width = int(img.shape[1] * ratio)
    return cv2.resize(img, (target_width, target_height))


def calculate_angle(a, b, c):
    """
    Calculates the angle at point b, between segment (a-b) and segment (b-c).
    Result in degrees: close to 180 = straight line (finger extended),
    close to 90 or less = sharp bend (finger curled).
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def is_finger_extended_by_angle(landmarks, mcp_id, pip_id, tip_id):
    """
    Returns True if a finger is extended, based on the bend angle
    at its middle joint (PIP). This is more reliable than comparing
    raw y-coordinates because it does not depend on hand orientation.
    """
    mcp = landmarks.landmark[mcp_id]
    pip = landmarks.landmark[pip_id]
    tip = landmarks.landmark[tip_id]

    angle = calculate_angle(mcp, pip, tip)
    ANGLE_THRESHOLD = 160

    return angle > ANGLE_THRESHOLD


def is_thumb_tucked(hand_landmarks):
    """
    Checks whether the thumb is tucked in against the palm (as in a
    natural POINTING gesture) rather than extended outward.
    Distance is measured relative to hand size so it works at any
    distance from the camera.
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    thumb_to_index_dist = np.sqrt(
        (thumb_tip.x - index_mcp.x) ** 2 + (thumb_tip.y - index_mcp.y) ** 2
    )
    hand_size = np.sqrt(
        (wrist.x - middle_mcp.x) ** 2 + (wrist.y - middle_mcp.y) ** 2
    )

    TUCK_RATIO = 0.55

    return thumb_to_index_dist < hand_size * TUCK_RATIO


def classify_gesture(hand_landmarks):
    """
    Classifies static hand gestures: THUMBS_UP, POINTING, PEACE.
    Falls back to NEUTRAL if nothing matches.
    """
    thumb_up = is_finger_extended_by_angle(
        hand_landmarks,
        mp_hands.HandLandmark.THUMB_CMC,
        mp_hands.HandLandmark.THUMB_MCP,
        mp_hands.HandLandmark.THUMB_TIP
    )
    index_up = is_finger_extended_by_angle(
        hand_landmarks,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP
    )
    middle_up = is_finger_extended_by_angle(
        hand_landmarks,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP
    )
    ring_up = is_finger_extended_by_angle(
        hand_landmarks,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_TIP
    )
    pinky_up = is_finger_extended_by_angle(
        hand_landmarks,
        mp_hands.HandLandmark.PINKY_MCP,
        mp_hands.HandLandmark.PINKY_PIP,
        mp_hands.HandLandmark.PINKY_TIP
    )

    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "THUMBS_UP"

    if index_up and is_thumb_tucked(hand_landmarks) and not middle_up and not ring_up and not pinky_up:
        return "POINTING"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    return "NEUTRAL"


def check_thinking_gesture(hand_landmarks, face_landmarks, frame_width, frame_height):
    """
    THINKING: index finger tip close to the nose/mouth area,
    with the middle finger folded down.
    """
    if not hand_landmarks or not face_landmarks:
        return False

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)

    nose_tip = face_landmarks.landmark[4]
    nose_x = int(nose_tip.x * frame_width)
    nose_y = int(nose_tip.y * frame_height)

    distance = np.sqrt((index_x - nose_x) ** 2 + (index_y - nose_y) ** 2)
    MAX_DISTANCE = 50

    y_middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    y_middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    is_middle_finger_down = y_middle_tip > y_middle_pip

    if distance < MAX_DISTANCE and is_middle_finger_down:
        return True

    return False

def check_smart_gesture(hand_landmarks, face_landmarks, frame_width, frame_height):
    """
    SMART: index finger tip close to either temple (left or right side).
    We check both temple landmarks so the gesture works with either hand.
    """
    if not hand_landmarks or not face_landmarks:
        return False

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)

    # 127 = one temple, 356 = the other temple
    temple_points = [127, 356]
    MAX_DISTANCE = 60

    for point_id in temple_points:
        temple = face_landmarks.landmark[point_id]
        temple_x = int(temple.x * frame_width)
        temple_y = int(temple.y * frame_height)

        distance = np.sqrt((index_x - temple_x) ** 2 + (index_y - temple_y) ** 2)

        if distance < MAX_DISTANCE:
            return True

    return False

CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

print("Gesture Tracker running. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    current_gesture = "NEUTRAL"
    hand_landmarks_data = None
    face_landmarks_data = None

    if hand_results.multi_hand_landmarks:
        hand_landmarks_data = hand_results.multi_hand_landmarks[0]

    if face_results.multi_face_landmarks:
        face_landmarks_data = face_results.multi_face_landmarks[0]

    if hand_landmarks_data:
        if face_landmarks_data and check_smart_gesture(
            hand_landmarks_data, face_landmarks_data, frame_width, frame_height
        ):
            current_gesture = "SMART"
        elif face_landmarks_data and check_thinking_gesture(
            hand_landmarks_data, face_landmarks_data, frame_width, frame_height
        ):
            current_gesture = "THINKING"
        else:
            current_gesture = classify_gesture(hand_landmarks_data)

        # Draw the hand skeleton (points + connecting lines) on the frame.
        # Delete this whole block if you don't want the overlay drawn.
        # mp_drawing.draw_landmarks(
        #     frame,
        #     hand_landmarks_data,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        #     mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
        # )

    gesture_image = load_and_resize_image(IMAGE_PATHS[current_gesture], frame_height)

    if gesture_image is not None:
        output_frame = np.concatenate((frame, gesture_image), axis=1)

        cv2.putText(
            output_frame,
            f"Gesture: {current_gesture.replace('_', ' ')}",
            (frame_width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    else:
        output_frame = frame
        cv2.putText(
            output_frame,
            "LOAD IMAGE FAILED - Check images/ folder!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow('Gesture & Image Pairing', output_frame)

    key = cv2.waitKey(5)
    if key == ord('q') or key == 27:
        break

hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
