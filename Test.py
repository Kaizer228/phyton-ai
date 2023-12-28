import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# OpenCV setup
cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark positions
            finger_tip_ids = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
            extended_fingers = sum(
                [1 for id in finger_tip_ids if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 1].y])

            # Draw circles on the fingertips
            for id in finger_tip_ids:
                cx, cy = int(hand_landmarks.landmark[id].x * frame.shape[1]), int(hand_landmarks.landmark[id].y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # Display the number of extended fingers
            cv2.putText(frame, f"Fingers: {extended_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Gesture Detector", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
