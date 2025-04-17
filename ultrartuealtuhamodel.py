import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import json
import onnxruntime as rt
from PIL import Image, ImageTk
from collections import deque
import threading
import queue
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModernSignTranslator:
    def __init__(self, root, config_path):
        self.root = root
        self.root.title("VisionSign Pro")
        self.root.geometry("1366x768")
        self.root.minsize(1024, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.colors = {
            'background': '#1A1A1A',
            'primary': '#2B95D6',
            'secondary': '#3A3A3A',
            'text': '#FFFFFF',
            'accent': '#00D1B2',
            'warning': '#FF6B6B'
        }
        
        self.setup_ui()
        self.load_config(config_path)
        self.init_video_processing()
        self.setup_bindings()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.control_panel = ttk.Frame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        
        self.video_panel = ttk.Frame(self.main_frame)
        self.video_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        self.create_device_selector()
        self.create_settings_controls()
        self.create_gesture_display()

    def configure_styles(self):
        self.style.configure('TFrame', background=self.colors['background'])
        self.style.configure(
            'Header.TLabel',
            font=('Helvetica', 14, 'bold'),
            foreground=self.colors['primary'],
            background=self.colors['background']
        )
        self.style.configure(
            'Primary.TButton',
            font=('Helvetica', 12),
            foreground=self.colors['text'],
            background=self.colors['primary'],
            borderwidth=0,
            padding=10
        )
        self.style.map(
            'Primary.TButton',
            background=[('active', '#2079B0'), ('disabled', '#505050')]
        )
        self.style.configure(
            'TCombobox',
            fieldbackground=self.colors['secondary'],
            background=self.colors['secondary'],
            foreground=self.colors['text']
        )

    def create_device_selector(self):
        header = ttk.Label(self.control_panel, text="Устройства ввода", style='Header.TLabel')
        header.pack(pady=10, anchor=tk.W)
        self.camera_selector = ttk.Combobox(self.control_panel, state='readonly', height=5)
        self.camera_selector.pack(fill=tk.X, pady=5)

    def create_settings_controls(self):
        settings_frame = ttk.Frame(self.control_panel)
        settings_frame.pack(fill=tk.X, pady=20)
        
        ttk.Label(settings_frame, text="Разрешение:", font=('Helvetica', 10)).pack(anchor=tk.W)
        self.resolution_selector = ttk.Combobox(
            settings_frame,
            values=['640x480', '1280x720', '1920x1080'],
            state='readonly'
        )
        self.resolution_selector.current(1)
        self.resolution_selector.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Частота кадров:", font=('Helvetica', 10)).pack(anchor=tk.W)
        self.fps_selector = ttk.Combobox(
            settings_frame,
            values=['15', '30', '60'],
            state='readonly'
        )
        self.fps_selector.current(1)
        self.fps_selector.pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(pady=15)
        self.start_btn = ttk.Button(
            btn_frame,
            text="Запустить",
            style='Primary.TButton',
            command=self.start_processing
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(
            btn_frame,
            text="Остановить",
            style='Primary.TButton',
            state=tk.DISABLED,
            command=self.stop_processing
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    def create_gesture_display(self):
        self.video_canvas = tk.Canvas(
            self.video_panel,
            bg=self.colors['background'],
            bd=0,
            highlightthickness=0
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.gesture_display = ttk.Label(
            self.video_panel,
            text="🖐️ Текущий жест: -",
            font=('Helvetica', 18, 'bold'),
            foreground=self.colors['accent'],
            background=self.colors['background']
        )
        self.gesture_display.pack(pady=10)
        
        self.history_display = ttk.Label(
            self.video_panel,
            text="📜 История: -",
            font=('Helvetica', 14),
            foreground=self.colors['text'],
            background=self.colors['background']
        )
        self.history_display.pack()

    def setup_bindings(self):
        self.root.bind('<Configure>', self.on_window_resize)
        self.root.bind('<Escape>', lambda e: self.on_close())

    def on_window_resize(self, event):
        if self.processing and not self.frame_queue.empty():
            self.update_gui()

    def init_video_processing(self):
        self.cap = None
        self.processing = False
        self.frame_queue = queue.Queue(maxsize=2)

    def load_config(self, config_path):
        required_keys = ['path_to_model', 'threshold', 'topk', 
                        'path_to_class_list', 'window_size', 'provider']
        try:
            with open(config_path) as f:
                self.config = json.load(f)
            
            for key in required_keys:
                if key not in self.config:
                    raise KeyError(f"Отсутствует ключ: {key}")
            
            self.config['window_size'] = int(self.config['window_size'])
            self.config['threshold'] = float(self.config['threshold'])
            self.config['topk'] = int(self.config['topk'])
            
            if not 0 < self.config['threshold'] <= 1:
                raise ValueError("Порог должен быть от 0 до 1")
            if self.config['window_size'] <= 0:
                raise ValueError("Размер окна должен быть > 0")
            if self.config['topk'] <= 0:
                raise ValueError("TopK должен быть > 0")

            self.init_model()
            self.detect_devices()
        except Exception as e:
            self.show_error(f"Ошибка конфигурации: {str(e)}")
            self.root.destroy()

    def init_model(self):
        try:
            options = rt.SessionOptions()
            options.intra_op_num_threads = 4
            options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
            
            available_providers = rt.get_available_providers()
            logging.info(f"Доступные провайдеры: {available_providers}")
            
            if self.config['provider'] not in available_providers:
                raise ValueError(f"Провайдер {self.config['provider']} недоступен")
            
            self.session = rt.InferenceSession(
                self.config['path_to_model'],
                sess_options=options,
                providers=[self.config['provider']]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.labels = self.load_labels()
            self.buffer = deque(maxlen=self.config['window_size'])
            self.gesture_history = deque(maxlen=5)
            logging.info("Модель успешно инициализирована")
        except Exception as e:
            self.show_error(f"Ошибка инициализации модели: {str(e)}")
            self.root.destroy()

    def load_labels(self):
        try:
            labels = {}
            with open(self.config['path_to_class_list'], 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        raise ValueError(f"Ошибка в строке {line_num}: неверный формат")
                    labels[int(parts[0])] = parts[1]
            return labels
        except Exception as e:
            self.show_error(f"Ошибка загрузки меток: {str(e)}")
            return {}

    def detect_devices(self):
        self.cameras = []
        index = 0
        while index < 4:
            try:
                cap = cv2.VideoCapture(index)
                if cap.isOpened() and cap.read()[0]:
                    self.cameras.append(f"Камера {index}")
                    logging.info(f"Обнаружена камера {index}")
                cap.release()
            except Exception as e:
                logging.error(f"Ошибка проверки камеры {index}: {str(e)}")
            index += 1
        
        self.camera_selector['values'] = self.cameras
        if self.cameras:
            self.camera_selector.current(0)
            self.start_btn.state(['!disabled'])
        else:
            self.start_btn.state(['disabled'])
            self.show_error("Не обнаружено доступных камер!")

    def start_processing(self):
        try:
            if not self.cameras:
                raise RuntimeError("Нет доступных камер")
            
            self.processing = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            camera_index = int(self.camera_selector.get().split()[-1])
            width, height = map(int, self.resolution_selector.get().split('x'))
            
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Не удалось подключиться к камере")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, int(self.fps_selector.get()))
            
            self.video_thread = threading.Thread(
                target=self.process_video, 
                daemon=True
            )
            self.video_thread.start()
            
            self.update_gui()
            logging.info("Обработка видео запущена")
        except Exception as e:
            self.show_error(f"Ошибка запуска: {str(e)}")
            self.stop_processing()

    def process_video(self):
        while self.processing:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Не удалось получить кадр с камеры")
                    continue
                
                processed = self.process_frame(frame)
                if not self.frame_queue.full():
                    self.frame_queue.put(processed, block=False)
            except Exception as e:
                logging.error(f"Ошибка обработки видео: {str(e)}")
                self.stop_processing()

    def process_frame(self, frame):
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Получен пустой кадр")
            
            # Предобработка кадра для модели
            model_frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
            self.buffer.append(model_frame)
            
            # Накопление буфера и выполнение предсказания
            if len(self.buffer) >= self.config['window_size']:
                clip = np.array(list(self.buffer)[-self.config['window_size']:])
                
                # Исправленное транспонирование осей (C, T, H, W)
                clip = np.transpose(clip, (3, 0, 1, 2))  # [C, T, H, W]
                
                # Добавление размерности батча
                outputs = self.session.run(None, {self.input_name: clip[None, ...]})
                probabilities = self.softmax(outputs[0][0])
                
                # Выбор топ-K предсказаний
                topk_indices = np.argsort(probabilities)[-self.config['topk']:][::-1]
                topk_confidences = probabilities[topk_indices]
                
                # Формирование результатов
                valid_gestures = []
                for idx, conf in zip(topk_indices, topk_confidences):
                    if conf > self.config['threshold']:
                        gesture = self.labels.get(idx, "Неизвестно")
                        valid_gestures.append(f"{gesture} ({conf:.2f})")
                
                if valid_gestures:
                    self.update_gesture_history(valid_gestures[0])
            
            # Подготовка кадра для отображения
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, f"FPS: {self.fps_selector.get()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame
        except Exception as e:
            logging.error(f"Ошибка обработки кадра: {str(e)}")
            return np.zeros((480, 640, 3), dtype=np.uint8) * 255

    def update_gesture_history(self, gesture):
        if not self.gesture_history or gesture != self.gesture_history[-1]:
            self.gesture_history.append(gesture)

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum()

    def update_gui(self):
        if self.processing:
            try:
                frame = self.frame_queue.get_nowait()
                img = Image.fromarray(frame)
                
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                img.thumbnail((canvas_width, canvas_height))
                
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_canvas.delete("all")
                self.video_canvas.create_image(
                    canvas_width//2, 
                    canvas_height//2, 
                    image=imgtk, 
                    anchor=tk.CENTER
                )
                self.video_canvas.image = imgtk
                
                current = "-"
                if self.gesture_history:
                    current = self.gesture_history[-1]
                self.gesture_display.config(
                    text=f"🖐️ Текущий жест: {current}",
                    foreground=self.colors['accent']
                )
                
                history_text = ", ".join(self.gesture_history) if self.gesture_history else "-"
                self.history_display.config(text=f"📜 История: {history_text}")
            
            except queue.Empty:
                pass
            
            self.root.after(1000//int(self.fps_selector.get()), self.update_gui)

    def show_error(self, message):
        self.gesture_display.config(
            text=f"❌ {message}",
            foreground=self.colors['warning']
        )
        messagebox.showerror("Ошибка", message)

    def stop_processing(self):
        self.processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.frame_queue.queue.clear()
        self.gesture_history.clear()
        logging.info("Обработка видео остановлена")

    def on_close(self):
        self.stop_processing()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernSignTranslator(root, r"C:\Users\rorov\питончик\easy_sign-main\configs\config.json")
    root.mainloop()