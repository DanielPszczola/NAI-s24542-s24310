"""
Symulacja lalki z serialu "Squid Game" - Young-Hee

Program symuluje zachowanie lalki z serialu "Squid Game". Wykorzystuje kamerę, aby:
- Namierzać twarz osoby w kadrze i nakładać na nią celownik w formie krzyżyka.
- Reagować na ruch w obrębie głowy, symulując strzał i wyświetlając komunikat "POW!".
- W przypadku podniesienia obu rąk nad głowę, system rozpoznaje poddanie się i wyświetla odpowiedni komunikat.

Instrukcja użycia:
1. Uruchom program w środowisku z dostępem do kamery.
2. Aby zakończyć działanie programu, naciśnij klawisz `q`.


Wymagania:
   - OpenCV,
   - Mediapipe

Autorzy: Michał Kaczmarek s24310, Daniel Pszczoła s24542

źródła danych do uczelnia algorytmu:
https://github.com/google-ai-edge/mediapipe
https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md:
    https://cocodataset.org/
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset
"""

import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def is_hands_up(hand_landmarks, frame_height):
    """
    Sprawdza, czy obie ręce są uniesione nad głowę.

    Arumenty:
        hand_landmarks: Lista punktów orientacyjnych wykrytych przez Mediapipe Pose.
        frame_height: Wysokość aktualnej klatki obrazu.

    Zwraca:
        bool: True, jeśli obie ręce są podniesione nad głowę. W przeciwnym razie False.
    """
    if not landmarks:
        return False

    left_hand = hand_landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]
    right_hand = hand_landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]
    head = hand_landmarks[mp_holistic.PoseLandmark.NOSE.value]

    return (left_hand.y < head.y and right_hand.y < head.y) and \
        (left_hand.y < 0.4 * frame_height and right_hand.y < 0.4 * frame_height)


cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        hands_up = is_hands_up(landmarks, frame.shape[0])

        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        nose = landmarks[mp_holistic.PoseLandmark.NOSE.value]
        x = int(nose.x * frame.shape[1])
        y = int(nose.y * frame.shape[0])
        cross_size = 20
        cv2.line(frame, (x - cross_size, y), (x + cross_size, y), (0, 255, 255), 2)
        cv2.line(frame, (x, y - cross_size), (x, y + cross_size), (0, 255, 255), 2)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        move_frame = cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), frame_gray)
        _, move_frame = cv2.threshold(move_frame, 25, 255, cv2.THRESH_BINARY)
        motion_in_frame = cv2.countNonZero(move_frame) > 5000

        if motion_in_frame and not hands_up:
            cv2.putText(frame, "POW!", (x - 40, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cross_size = 40
            cv2.line(frame, (x - cross_size, y), (x + cross_size, y), (0, 0, 255), 4)
            cv2.line(frame, (x, y - cross_size), (x, y + cross_size), (0, 0, 255), 4)

        elif hands_up:
            cv2.putText(frame, "Poddaje sie!", (x - 60, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    prev_frame = frame.copy()

    cv2.imshow("Squid Game Doll Young-Hee", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
