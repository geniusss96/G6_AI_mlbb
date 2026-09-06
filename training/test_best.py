from ultralytics import YOLO
import glob
import os
import cv2

# Указываем абсолютный путь к вашим обученным весам
# (Используем точный путь в текущей рабочей папке)
weights_path = r"c:\Users\user\Desktop\скрипт(mlbb)\runs\detect\mlbb_training\poc_run\weights\best.pt"

if not os.path.exists(weights_path):
    print(f"[ERROR] Не найден файл весов по пути: {weights_path}")
    exit(1)

print(f"[*] Загружаем веса: {weights_path}")
model = YOLO(weights_path)

# Берем любую картинку из папки test для проверки
test_dir = r"c:\Users\user\Desktop\скрипт(mlbb)\dataset_raw\MLBB_Claude_Bot.v2i.yolov8\test\images\*.jpg"
test_images = glob.glob(test_dir)

if test_images:
    sample_image = test_images[0]
    print(f"[*] Проверяем картинку: {os.path.basename(sample_image)}")
    
    # Запускаем предсказание
    results = model(sample_image, conf=0.4)
    
    # Отрисовываем Bounding Boxes на картинке
    res_plotted = results[0].plot()
    
    # Сохраняем результат для надежности
    cv2.imwrite("final_poc_test.jpg", res_plotted)
    print("[*] Картинка сохранена как final_poc_test.jpg")
    
    # Выводим на экран и ждем нажатия клавиши
    print("[*] Готово! Открываю окно. Нажмите ЛЮБУЮ КЛАВИШУ на картинке, чтобы закрыть.")
    cv2.imshow("YOLOv8 Test Result", res_plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"[!] Не найдены тестовые картинки по пути: {test_dir}")
