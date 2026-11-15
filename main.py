import cv2
import pickle
import numpy as np
import mediapipe as mp

# --- Video capture ---
cap = cv2.VideoCapture(0)

# --- Load trained model (saved as {'model': ...}) ---
mdl_dict = pickle.load(open('./model.p', 'rb'))
model = mdl_dict['model']

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# For live video use static_image_mode=False; track up to 2 hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Label map
dataset_labels = {
    0: 'Thumbs Up', 1: 'Love', 2: 'Peace or Victory Sign', 3: 'Ok Sign',
    4: 'Vulcan Salute', 5: 'Rock On', 6: 'Fingers Crossed', 7: 'Left', 8: 'Right'
}

stop_key = ord('q')

# --- helper functions ---

def hand_xy_feats_norm(hand_lms):
    """
    Build 42-length feature vector (x,y for 21 landmarks) normalized by per-hand min x/y.
    """
    xs = [lm.x for lm in hand_lms.landmark]
    ys = [lm.y for lm in hand_lms.landmark]
    minx, miny = min(xs), min(ys)
    feats = []
    for lm in hand_lms.landmark:
        feats.extend([lm.x - minx, lm.y - miny])
    return feats  # 42

def zero_vec(n):
    return [0.0] * n

def rect_from_hands(hands_lms, width, height):
    """
    Compute a bounding rectangle covering all visible hands.
    """
    if not hands_lms:
        return 0, 0, width, height
    xs = []
    ys = []
    for hlms in hands_lms:
        xs.extend([lm.x for lm in hlms.landmark])
        ys.extend([lm.y for lm in hlms.landmark])
    minx, miny = min(xs), min(ys)
    maxx, maxy = max(xs), max(ys)
    margin = 30
    x1 = max(0, int(minx * width) - margin)
    y1 = max(0, int(miny * height) - margin)
    x2 = min(width, int(maxx * width) + margin)
    y2 = min(height, int(maxy * height) + margin)
    return x1, y1, x2, y2

# Try to read expected feature length from the model (scikit-learn supports this)
expected_feats = getattr(model, "n_features_in_", None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_character = None

    if results.multi_hand_landmarks:
        # Draw landmarks
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        # Identify left vs right using handedness
        left_lms = None
        right_lms = None
        if results.multi_handedness:
            for lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handed.classification[0].label  # 'Left' or 'Right'
                if label == 'Left':
                    left_lms = lms
                elif label == 'Right':
                    right_lms = lms

        # Build 84-dim vector: Left 42 + Right 42 (x,y) per hand with per-hand min normalization
        left_vec = hand_xy_feats_norm(left_lms) if left_lms is not None else zero_vec(42)
        right_vec = hand_xy_feats_norm(right_lms) if right_lms is not None else zero_vec(42)
        data_vec = left_vec + right_vec  # 84 features

        # Safety: if the trained model expects a different length (e.g., 42), adapt safely.
        if expected_feats is not None and expected_feats != len(data_vec):
            # If model expects 42, pick the non-zero hand (prefer Right if present)
            if expected_feats == 42:
                use = right_vec if any(right_vec) else left_vec
                data_vec = use[:42] + zero_vec(max(0, 42 - len(use)))
            else:
                # If some other length, raise a clear error to avoid silent misclassification
                raise ValueError(f"Built {len(data_vec)} features but model expects {expected_feats}.")

        data_vec = np.asarray(data_vec, dtype=np.float32)

        # Predict
        prediction = model.predict([data_vec])
        predicted_character = dataset_labels.get(int(prediction[0]), "Unknown")

        # Draw one bounding rectangle around all detected hands
        x1, y1, x2, y2 = rect_from_hands(results.multi_hand_landmarks, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # UI: hint to quit
    cv2.putText(frame, "Press 'Q' to stop", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == stop_key:
        break

cap.release()
cv2.destroyAllWindows()
