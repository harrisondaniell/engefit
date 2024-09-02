import cv2
import mediapipe as mp
import math

video = cv2.VideoCapture("polichinelos1.mp4")  


pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.7, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

contador = 0
check = True

while True:
    success, img = video.read()
    if not success:
        print("Falha ao capturar o vídeo. Saindo...")
        break  # Sai do loop se não conseguir capturar um frame

    # Pré-processamento da imagem
    img = cv2.GaussianBlur(img, (5, 5), 0)

    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)

    h, w, _ = img.shape

    if points:
        # Extrair coordenadas dos landmarks
        peDY = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h)
        peDX = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].x * w)
        peEY = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].y * h)
        peEX = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].x * w)
        moDY = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y * h)
        moDX = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x * w)
        moEY = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y * h)
        moEX = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x * w)

        # Calcular distâncias
        distMO = math.hypot(moDX - moEX, moDY - moEY)
        distPE = math.hypot(peDX - peEX, peDY - peEY)

        print(f'maos {distMO} pes {distPE}')

        # Ajustar os limites para melhorar a detecção
        LIMITE_DISTANCIA_MAO = 150
        LIMITE_DISTANCIA_PE = 150

        if check and distMO <= LIMITE_DISTANCIA_MAO and distPE >= LIMITE_DISTANCIA_PE:
            contador += 1
            check = False

        if distMO > LIMITE_DISTANCIA_MAO and distPE < LIMITE_DISTANCIA_PE:
            check = True

        texto = f'QTD {contador}'
        cv2.rectangle(img, (20, 240), (280, 120), (255, 0, 0), -1)
        cv2.putText(img, texto, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    cv2.imshow('Resultado', img)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
