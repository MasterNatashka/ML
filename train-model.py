# https://docs.ultralytics.com/ru/modes/train
from ultralytics import YOLO

# Альтернативно тренировку можно запустить из консоли командой
# yolo detect train data=datasets\puddle\data.yaml model=weights\yolov8n.pt name=puddle epochs=1 batch=32 workers=1 imgsz=640

# Предотвращение ошибки многопоточной обработки
if __name__ == "__main__":
    # Инициализация предобученной модели
    model = YOLO("./weights/yolov8n.pt")
    # Тренировка на подготовленном наборе данных
    model.train(
        data="./datasets/puddle/data.yaml",
        # Имя тренировочного прогона. Результаты будут сохранены в папке
        name="puddle",
        # Количество эпох обучения https://www.ultralytics.com/ru/glossary/epoch
        epochs=5,
        # При выставлении больших значений batch и workers
        # могут возникать ошибки из-за нехватки памяти GPU
        # Размер партии https://www.ultralytics.com/ru/glossary/batch-size
        batch=50,
        # Количество рабочих потоков для загрузки данных
        workers=1,
        # Целевой размер изображения для обучения
        imgsz=640
    )
    # Валидация на тренировочных данных
    model.val()
