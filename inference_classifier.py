import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('/home/skullboxml/ISL/model.p', 'rb'))
model = model_dict['model']
max_features = model_dict.get('max_features', 84)  

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

alphabet = [chr(i) for i in range(65, 91) if chr(i) != 'Q'] 
labels_dict = {i: alphabet[i] for i in range(len(alphabet))}

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        continue 

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        if len(data_aux) < max_features:
            data_aux = np.pad(data_aux, (0, max_features - len(data_aux)), mode="constant")

        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

       
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict.get(int(prediction[0]), "?")  

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
