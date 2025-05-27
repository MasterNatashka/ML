# https://docs.ultralytics.com/ru/usage/python/
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Инициализация модели
model = YOLO("./weights/yolov8l.pt")

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ok, frame = cap.read()
    # Результаты обнаружения
    # stream=True позволяет более эффективно использовать память, 
    # что важно для обработки длинных видеороликов или больших наборов данных.
    # По умолчанию conf=0.25 - минимальный порог уверенности обнаружения,
    # можно увеличить для уменьшения количества ложных срабатываний.
    # Подробнее https://docs.ultralytics.com/ru/modes/predict
    results = model.predict(frame, stream=True, conf=0.5)
    for res in results:
        for box in res.boxes:
            # Рамка обнаруженного объекта
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # Класс обнаруженного объекта
            object_class = model.names[int(box.cls[0])]
            # Уверенность обнаружения (от 0 до 1)
            conf = box.conf[0]
            cvzone.putTextRect(
                frame,
                f'{object_class} {conf:.2f}',
                (max(0, x1), max(30, y1)),
                scale=1,
                thickness=1
            )
            # :.2f - форматирование числа, где 2 - количество знаков после запятой
    # Вывод кадра
    cv2.imshow("Image", frame)
