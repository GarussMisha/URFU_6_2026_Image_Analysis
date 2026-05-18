import cv2
import time
import os
import threading
import traceback
from queue import Queue
from config import ( # Импорт настроек из файла конфигурации
    RTSP_URL,        # URL RTSP-камеры
    OUTPUT_DIR,      # Директория для снимков
    MIN_AREA,        # Мин. площадь движения (пиксели)
    THRESHOLD,       # Порог различия кадров (0-255)
    MOTION_DURATION, # Мин. длительность движения (кадры)
    COOLDOWN_SECONDS,# Время ожидания после снимка (сек)
    FRAME_SKIP       # Пропуск кадров между анализами
)

class AsyncSnapshotSaver:
    """Асинхронный поток для сохранения скриншотов"""
    
    def __init__(self, output_dir, resize_factor=0.5): # Директория для снимков, фактор уменьшения размера кадра
        self.queue = Queue()               # Очередь для кадров
        self.output_dir = output_dir       # Папка сохранения
        self.resize_factor = resize_factor # Уменьшение размера на 50%
        self.running = True                # Флаг работы потока
        self.saved_count = 0               # Счётчик сохранённых снимков
        self.error_count = 0               # Счётчик ошибок
        self.max_retries = 3               # Максимум попыток сохранения
        self.retry_delay = 0.1             # Задержка между попытками
        
        # Даём время на инициализацию
        time.sleep(0.5)
        self.thread = threading.Thread(target=self._save_loop, daemon=True)
        self.thread.start()
        print("🔄 Фоновый поток сохранения запущен")
    
    def _save_loop(self):
        """Фоновый поток для сохранения"""
        while self.running:
            try:
                # Берём кадр из очереди
                frame, timestamp = self.queue.get(timeout=2)
                
                if frame is not None:
                    # Копируем кадр критически важно!
                    frame_copy = frame.copy()
                    
                    # Проверяем валидность кадра
                    if frame_copy.size == 0:
                        print(f"⚠️ Пустой кадр получен")
                        continue
                    
                    # Уменьшаем размер для быстрой записи
                    if self.resize_factor < 1.0:
                        h, w = frame_copy.shape[:2]
                        frame_copy = cv2.resize(frame_copy, (int(w * self.resize_factor), int(h * self.resize_factor)))
                    
                    filename = f"{self.output_dir}/motion_{self.saved_count:03d}.jpg"
                    
                    # Пробуем сохранить с ретраями
                    success = False
                    for attempt in range(self.max_retries):
                        try:
                            success = cv2.imwrite(filename, frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success:
                                break
                            time.sleep(self.retry_delay)
                        except Exception as e:
                            print(f"⚠️ Попытка {attempt + 1} неудачна: {e}")
                            time.sleep(self.retry_delay)
                    
                    if success:
                        self.saved_count += 1
                        print(f"📸 Сохранено: {filename}")
                    else:
                        print(f"⚠️ Ошибка записи файла (после {self.max_retries} попыток): {filename}")
                        print(f"   Размер кадра: {frame_copy.shape}")
                        self.error_count += 1
                else:
                    print("⚠️ Получен None кадр")
                    
            except Exception as e:
                # Проверяем, не является ли это Empty (таймаут очереди)
                if "Empty" in str(type(e).__name__) or "Empty" in str(e):
                    # Это просто таймаут - ничего не делаем и продолжаем цикл
                    continue
                print(f"⚠️ Ошибка сохранения: {str(e)}")
                traceback.print_exc()
                self.error_count += 1
    
    def save_frame(self, frame):
        """Добавить кадр в очередь на сохранение"""
        if frame is not None and self.running:
            # КРИТИЧНО: Копируем кадр перед отправкой
            frame_copy = frame.copy()
            self.queue.put((frame_copy, time.time()))
    
    def stop(self):
        """Остановить поток"""
        self.running = False
        self.thread.join(timeout=5)
        print(f"📊 Статистика: сохранено={self.saved_count}, ошибок={self.error_count}")


class MotionDetector:
    """Класс для детекции движения на видеоходе и сохранения снимков."""
    def __init__(self):
        self.prev_frame = None                    # Предыдущий кадр (для сравнения)
        self.motion_frames = 0                    # Счётчик кадров с движением
        self.min_motion_frames = MOTION_DURATION  # Минимум кадров
        self.cooldown_active = False              # Флаг cooldown
        self.cooldown_timer = 0                   # Таймер cooldown
        self.snapshot_counter = 0                 # Счётчик снимков
        self.saver = None                         # Ссылка на сохранитель
    
    def setup(self):
        """Создаём директорию и инициализируем сохранитель"""
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        print(f"📁 Директория: {OUTPUT_DIR}")
        print(f"⚙️ Настройки:")
        print(f"   Мин. площадь: {MIN_AREA}")
        print(f"   Порог: {THRESHOLD}")
        print(f"   Мин. длительность: {MOTION_DURATION} кадров")
        print(f"   Cooldown: {COOLDOWN_SECONDS} сек")
        
        # Инициализируем асинхронный сохранитель
        self.saver = AsyncSnapshotSaver(OUTPUT_DIR, resize_factor=0.5)
    
    def _reconnect(self, cap):
        """Повторное подключение к камере"""
        max_retries = 10
        retry_delay = 5
        
        for attempt in range(max_retries):
            print(f"🔄 Попытка переподключения {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay)
            
            # Освобождаем старый поток
            cap.release()
            
            # Подключаемся заново
            cap = cv2.VideoCapture(RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
            
            if cap.isOpened():
                print("✅ Переподключение успешно!")
                # Сбрасываем фон при переподключении
                self.prev_frame = None
                return cap
        
        print("❌ Не удалось переподключиться после", max_retries, "попыток")
        return None
    
    def detect_motion(self):
        print("📹 Подключение к камере...")
        cap = cv2.VideoCapture(RTSP_URL)
        
        # Устанавливаем буфер RTSP
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 10) # Увеличение буфера RTSP
        
        if not cap.isOpened():
            print("❌ Ошибка подключения к камере !")
            return
        
        print("✅ Подключение к камере успешно! Начало детекции...")
        print("   Нажмите 'q' на окне камеры для выхода")
        
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"⚠️ Ошибка чтения кадра ({consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    # Пробуем переподключиться
                    new_cap = self._reconnect(cap)
                    if new_cap:
                        cap = new_cap
                        consecutive_failures = 0
                        continue
                    else:
                        break
                continue
            else:
                consecutive_failures = 0
            
            frame_count += 1
            
            # Cooldown
            if self.cooldown_active:
                if time.time() - self.cooldown_timer > COOLDOWN_SECONDS:
                    self.cooldown_active = False
                    print("   Cooldown завершён")
                else:
                    continue
            
            # Пропускаем кадры
            if frame_count % FRAME_SKIP != 0:
                continue
            
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Инициализация
            if self.prev_frame is None:
                self.prev_frame = gray.copy()
                print("🔄 Инициализация фона...")
                continue
            
            # Вычисляем разницу
            diff = cv2.absdiff(self.prev_frame, gray)
            _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # Размываем и находим контуры
            thresh = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Анализируем контуры
            total_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > MIN_AREA:
                    total_area += area
            
            if total_area > 0:
                self.motion_frames += 1
                print(f"   Движение: {total_area} пикселей (кадров: {self.motion_frames})")
            else:
                if self.motion_frames > 0:
                    print(f"   Движение прекратилось (было {self.motion_frames} кадров)")
                self.motion_frames = 0
            
            # Если накопилось достаточно кадров движения
            if self.motion_frames >= self.min_motion_frames:
                self.snapshot_counter += 1
                
                # АСИНХРОННО добавляем в очередь на сохранение
                self.saver.save_frame(frame)
                print(f"🔴 ДВИЖЕНИЕ! Отправлено на сохранение")
                
                # Активируем cooldown
                self.cooldown_active = True
                self.cooldown_timer = time.time()
                self.motion_frames = 0
            
            # Обновляем предыдущий кадр
            self.prev_frame = gray.copy()
            
            # Показываем результат
            cv2.imshow("Motion Detection", frame)
            cv2.imshow("Threshold", thresh)
            
            # Выход по 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Останавливаем фоновый поток
        if self.saver:
            self.saver.stop()
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📸 Всего сохранено: {self.snapshot_counter} снимков")

if __name__ == "__main__":
    detector = MotionDetector()
    detector.setup()
    try:
        detector.detect_motion()
    except KeyboardInterrupt:
        print("\n⚠️ Программа прервана пользователем")
        if detector.saver:
            detector.saver.stop()
        print("👋 До свидания!")