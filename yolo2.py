import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
from PIL import Image, ImageDraw, ImageFont
import os


class SquatTrainer:
    def __init__(self, model_name='yolo11s-pose.pt', conf_threshold=0.7):
        """
        Initialize squat tracker

        Args:
            model_name: YOLO model name
            conf_threshold: confidence threshold for detection
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Key points for squat analysis
        self.body_points = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        # Ideal angles for squats (in degrees)
        self.ideal_knee_angle = 90
        self.ideal_hip_angle = 45

        # Current state
        self.squat_depth = 0
        self.feedback_messages = []
        self.knee_feedback = []
        self.rep_count = 0
        self.is_squatting = False
        self.min_knee_angle = 180
        self.view_angle = "unknown"  # front, side, or unknown

        # Colors for feedback
        self.colors = {
            'good': (0, 255, 0),
            'warning': (0, 255, 255),
            'bad': (0, 0, 255),
            'info': (255, 255, 255)
        }

        # Загрузка шрифта для русского текста
        self.font_path = self.get_font_path()
        self.pil_font_large = ImageFont.truetype(self.font_path, 24) if self.font_path else None
        self.pil_font_medium = ImageFont.truetype(self.font_path, 20) if self.font_path else None
        self.pil_font_small = ImageFont.truetype(self.font_path, 16) if self.font_path else None

    def get_font_path(self):
        """Поиск подходящего шрифта для русского текста"""
        possible_fonts = [
            'arial.ttf', 'arialbd.ttf', 'DejaVuSans.ttf',
            'LiberationSans-Regular.ttf', 'Roboto-Regular.ttf'
        ]

        font_paths = [
            '/usr/share/fonts/truetype/freefont/',
            '/usr/share/fonts/truetype/dejavu/',
            '/usr/share/fonts/truetype/liberation/',
            'C:/Windows/Fonts/',
            '/Library/Fonts/'
        ]

        for font_path in font_paths:
            for font_name in possible_fonts:
                full_path = os.path.join(font_path, font_name)
                if os.path.exists(full_path):
                    return full_path

        print("Warning: Russian font not found, using default")
        return None

    def put_ru_text(self, image, text, position, font_size='medium', color=(255, 255, 255)):
        """Функция для отображения русского текста на изображении"""
        try:
            if font_size == 'large' and self.pil_font_large:
                font = self.pil_font_large
            elif font_size == 'medium' and self.pil_font_medium:
                font = self.pil_font_medium
            elif font_size == 'small' and self.pil_font_small:
                font = self.pil_font_small
            else:
                # Резервный вариант без русского текста
                cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                return image

            # Конвертируем изображение OpenCV в PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # Рисуем текст
            draw.text(position, text, font=font, fill=color)

            # Конвертируем обратно в OpenCV
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error displaying text: {e}")
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return image

    def determine_view_angle(self, keypoints):
        """Determine if the view is front, side, or unknown based on shoulder and hip positions"""
        try:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]

            # Calculate horizontal distances
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            hip_width = abs(left_hip[0] - right_hip[0])

            # If shoulders and hips are wide (visible from front)
            if shoulder_width > 100 and hip_width > 80:
                return "фронтальный"
            # If shoulders and hips are narrow (visible from side)
            elif shoulder_width < 60 and hip_width < 50:
                return "профиль"
            else:
                return "неопределен"
        except:
            return "неопределен"

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points (b - vertex)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_torso_angle(self, shoulder_center, hip_center, ankle_center):
        """Calculate torso angle with improved algorithm for profile view"""
        try:
            # Calculate vertical line from hips
            vertical_point = [hip_center[0], hip_center[1] - 100]  # Point 100px above hips

            # Calculate angle between vertical line and torso line (shoulders to hips)
            dx_vertical = hip_center[0] - vertical_point[0]
            dy_vertical = hip_center[1] - vertical_point[1]

            dx_torso = shoulder_center[0] - hip_center[0]
            dy_torso = shoulder_center[1] - hip_center[1]

            # Calculate dot product
            dot_product = dx_vertical * dx_torso + dy_vertical * dy_torso

            # Calculate magnitudes
            mag_vertical = math.sqrt(dx_vertical ** 2 + dy_vertical ** 2)
            mag_torso = math.sqrt(dx_torso ** 2 + dy_torso ** 2)

            # Calculate angle in radians, then convert to degrees
            angle_rad = math.acos(dot_product / (mag_vertical * mag_torso))
            angle_deg = math.degrees(angle_rad)

            # Adjust angle based on shoulder position relative to hips
            if shoulder_center[0] < hip_center[0]:  # Shoulders are to the left of hips
                angle_deg = -angle_deg

            return abs(angle_deg)
        except:
            return 0

    def analyze_knee_position(self, keypoints):
        """Analyze knee position relative to feet and hips"""
        knee_feedback = []

        try:
            left_hip = [keypoints[11][0], keypoints[11][1]]
            right_hip = [keypoints[12][0], keypoints[12][1]]
            left_knee = [keypoints[13][0], keypoints[13][1]]
            right_knee = [keypoints[14][0], keypoints[14][1]]
            left_ankle = [keypoints[15][0], keypoints[15][1]]
            right_ankle = [keypoints[16][0], keypoints[16][1]]

            # Check if knees are tracking over toes
            left_knee_over_toe = abs(left_knee[0] - left_ankle[0]) < 100
            right_knee_over_toe = abs(right_knee[0] - right_ankle[0]) < 100

            if left_knee_over_toe and right_knee_over_toe:
                knee_feedback.append("Колени над стопами: Хорошо")
            else:
                knee_feedback.append("Колени выходят за стопы!")

            # Check knee symmetry
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            knee_diff = abs(left_knee_angle - right_knee_angle)

            if knee_diff > 15:
                knee_feedback.append("Колени несимметричны!")
            else:
                knee_feedback.append("Симметрия коленей: Хорошо")

            # Check knee stability (movement during squat)
            if left_knee_angle < 100 or right_knee_angle < 100:  # In squat position
                left_knee_stable = abs(left_knee[0] - left_hip[0]) < 100
                right_knee_stable = abs(right_knee[0] - right_hip[0]) < 100

                if left_knee_stable and right_knee_stable:
                    knee_feedback.append("Стабильность коленей: Хорошо")
                else:
                    knee_feedback.append("Колени шатаются - стабилизируйте!")

            # Check knee alignment with hips
            left_alignment = abs(left_knee[0] - left_hip[0]) < 150
            right_alignment = abs(right_knee[0] - right_hip[0]) < 150

            if left_alignment and right_alignment:
                knee_feedback.append("Выравнивание коленей: Хорошо")
            else:
                knee_feedback.append("Колени заваливаются внутрь!")

        except Exception as e:
            knee_feedback.append("Анализ коленей: Недостаточно данных")

        return knee_feedback

    def analyze_squat_form(self, keypoints):
        """Analyze squat form and provide feedback"""
        feedback = []
        warnings = []

        try:
            # Determine view angle for better analysis
            self.view_angle = self.determine_view_angle(keypoints)

            # Get keypoint coordinates
            left_hip = [keypoints[11][0], keypoints[11][1]]
            right_hip = [keypoints[12][0], keypoints[12][1]]
            left_knee = [keypoints[13][0], keypoints[13][1]]
            right_knee = [keypoints[14][0], keypoints[14][1]]
            left_ankle = [keypoints[15][0], keypoints[15][1]]
            right_ankle = [keypoints[16][0], keypoints[16][1]]
            left_shoulder = [keypoints[5][0], keypoints[5][1]]
            right_shoulder = [keypoints[6][0], keypoints[6][1]]

            # Calculate center points for symmetry
            hip_center = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
            knee_center = [(left_knee[0] + right_knee[0]) / 2, (left_knee[1] + right_knee[1]) / 2]
            shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
            ankle_center = [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]

            # Calculate angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            knee_angle = (left_knee_angle + right_knee_angle) / 2

            # Use improved torso angle calculation
            torso_angle = self.calculate_torso_angle(shoulder_center, hip_center, ankle_center)

            # Analyze knee position
            self.knee_feedback = self.analyze_knee_position(keypoints)

            # Update minimum knee angle for depth tracking
            if knee_angle < self.min_knee_angle:
                self.min_knee_angle = knee_angle

            # Analyze squat form
            knee_diff = abs(left_knee_angle - right_knee_angle)

            # 1. Squat depth analysis
            squat_percentage = max(0, min(100, (180 - knee_angle) / (180 - self.ideal_knee_angle) * 100))
            self.squat_depth = squat_percentage

            # 2. Knee analysis
            if knee_angle < 60:
                warnings.append("Слишком глубоко! Риск травмы")
            elif knee_angle > 120:
                feedback.append("Опуститесь глубже для полного диапазона")
            elif 80 <= knee_angle <= 100:
                feedback.append("Достигнута идеальная глубина")
            else:
                feedback.append("Хорошая глубина")

            if knee_diff > 15:
                warnings.append("Колени несимметричны!")

            # 3. Back analysis with improved calculation
            # More lenient thresholds for back angle
            if torso_angle > 400:
                warnings.append(f"Спина слишком наклонена! ")
            elif torso_angle < 5:
                warnings.append("Слишком прямо - напрягите корпус")
            else:
                feedback.append(f"Хорошее положение спины ")

            # 4. Repetition tracking
            if knee_angle < 100 and not self.is_squatting:
                self.is_squatting = True
            elif knee_angle > 160 and self.is_squatting:
                self.is_squatting = False
                self.rep_count += 1
                self.min_knee_angle = 180
                feedback.append(f"Повторение {self.rep_count} завершено!")

            # Add view information
            feedback.append(f"Ракурс: {self.view_angle}")

            return feedback, warnings, {
                'knee_angle': knee_angle,
                'torso_angle': torso_angle,
                'squat_depth': squat_percentage
            }

        except Exception as e:
            return ["Не все точки тела видны"], [], {}

    def draw_body_points(self, result, frame):
        """Draw body points with additional information"""
        if result.keypoints is None:
            return frame

        annotated_frame = frame.copy()
        keypoints = result.keypoints.data.cpu().numpy()

        for person_kpts in keypoints:
            # Analyze pose
            feedback, warnings, angles = self.analyze_squat_form(person_kpts)
            self.feedback_messages = feedback + warnings

            # Draw body points
            for point_id in self.body_points:
                if point_id < len(person_kpts) and person_kpts[point_id][2] > 0.3:
                    x, y = int(person_kpts[point_id][0]), int(person_kpts[point_id][1])

                    # Point color based on joint type
                    if point_id in [13, 14]:  # Knees
                        color = self.colors['good'] if angles.get('knee_angle', 0) > 80 and angles.get('knee_angle',
                                                                                                       0) < 100 else \
                            self.colors['warning']
                    elif point_id in [11, 12]:  # Hips
                        color = self.colors['good']
                    else:
                        color = self.colors['info']

                    cv2.circle(annotated_frame, (x, y), 8, color, -1)
                    cv2.circle(annotated_frame, (x, y), 10, (255, 255, 255), 2)

        return annotated_frame

    def add_fitness_info(self, frame, result):
        """Add enlarged information panel for squat tracker"""
        h, w = frame.shape[:2]

        # Main info panel - enlarged
        panel_width = 550
        panel_height = 600
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Info text with Russian language and larger font
        y_offset = 40
        line_height = 30

        frame = self.put_ru_text(frame, "ФИТНЕС-ТРЕКЕР: ПРИСЕДАНИЯ", (20, y_offset),
                                 font_size='large', color=self.colors['info'])
        y_offset += line_height + 10

        frame = self.put_ru_text(frame, f"Повторения: {self.rep_count}", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height

        frame = self.put_ru_text(frame, f"Глубина приседания: {self.squat_depth:.1f}%", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height

        frame = self.put_ru_text(frame, f"FPS: {self.fps:.1f}", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height

        frame = self.put_ru_text(frame, f"Ракурс: {self.view_angle}", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height + 5

        # Form Feedback
        frame = self.put_ru_text(frame, "АНАЛИЗ ФОРМЫ:", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height

        for i, message in enumerate(self.feedback_messages[:4]):
            color = self.colors['warning'] if "!" in message or "Too" in message else self.colors['good']
            frame = self.put_ru_text(frame, f"- {message}", (30, y_offset),
                                     font_size='small', color=color)
            y_offset += line_height - 5

        # Knee Analysis Section
        frame = self.put_ru_text(frame, "АНАЛИЗ КОЛЕНЕЙ:", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height

        for i, message in enumerate(self.knee_feedback[:4]):
            color = self.colors['warning'] if "!" in message or "Watch" in message or "collapsing" in message else \
            self.colors['good']
            frame = self.put_ru_text(frame, f"- {message}", (30, y_offset),
                                     font_size='small', color=color)
            y_offset += line_height - 5

        # Technique Tips
        frame = self.put_ru_text(frame, "СОВЕТЫ ПО ТЕХНИКЕ:", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height

        tips = [
            "Держите грудь поднятой и спину прямой",
            "Колени должны быть над стопами",
            "Опускайтесь до параллели бедер с полом",
            "Поднимайтесь через пятки"
        ]

        for i, tip in enumerate(tips[:2]):
            frame = self.put_ru_text(frame, f"- {tip}", (30, y_offset),
                                     font_size='small', color=self.colors['info'])
            y_offset += line_height - 8

        # Squat depth indicator
        bar_x, bar_y = w - 150, 50
        bar_width, bar_height = 30, 200
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (100, 100, 100), -1)

        # Green zone (ideal depth)
        ideal_start = bar_y + bar_height * 0.6
        ideal_end = bar_y + bar_height * 0.4
        cv2.rectangle(frame, (bar_x, int(ideal_start)), (bar_x + bar_width, int(ideal_end)),
                      (0, 200, 0), -1)

        # Current depth
        current_height = bar_y + bar_height * (1 - self.squat_depth / 100)
        cv2.rectangle(frame, (bar_x, bar_y + bar_height),
                      (bar_x + bar_width, int(current_height)),
                      (255, 255, 255), -1)

        frame = self.put_ru_text(frame, "Глубина", (bar_x - 10, bar_y + bar_height + 20),
                                 font_size='small', color=self.colors['info'])

        # Control hints
        frame = self.put_ru_text(frame, "Управление: Q-выход, R-сброс, P-пауза, +/- чувствительность", (10, h - 10),
                                 font_size='small', color=self.colors['info'])

        return frame

    def calculate_fps(self):
        """Calculate FPS"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        return fps

    def reset_counter(self):
        """Reset repetition counter"""
        self.rep_count = 0
        self.min_knee_angle = 180

    def process_frame(self, frame):
        """Process single frame"""
        # Perform prediction
        results = self.model(
            frame,
            conf=self.conf_threshold,
            imgsz=640,
            iou=0.5,
            max_det=1,
            verbose=False
        )

        # Visualize results
        annotated_frame = self.draw_body_points(results[0], frame)

        # Calculate FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.fps = self.calculate_fps()

        # Add information to frame
        annotated_frame = self.add_fitness_info(annotated_frame, results[0])

        return annotated_frame, results[0]


def main():
    # Initialize squat tracker
    trainer = SquatTrainer(
        model_name='yolo11s-pose.pt',
        conf_threshold=0.6
    )

    # Load video file
    video_path = 'squat_video.mp4'  # Change this to your video file path
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Видео загружено: {video_path}")
    print(f"Свойства видео: {width}x{height}, {fps:.2f} FPS, {total_frames} кадров")
    print("Запущен улучшенный анализ приседаний!")
    print("Функции:")
    print("- Улучшенное вычисление углов")
    print("- Детальный анализ положения коленей")
    print("- Советы по технике на русском языке")
    print("- Управление: Q-выход, R-сброс, P-пауза, +/- чувствительность")

    # Video writer to save output
    output_path = 'улучшенный_анализ_приседаний.mp4'

    # Try different codecs for better compatibility
    codecs = ['mp4v', 'X264', 'avc1', 'MJPG']
    fourcc = None
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        if fourcc != -1:
            print(f"Используется кодек: {codec}")
            break

    if fourcc == -1 or fourcc is None:
        print("Предупреждение: Подходящий кодек не найден, используется стандартный")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter with the same parameters as input video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Ошибка: Не удалось открыть выходной видеофайл {output_path}")
        cap.release()
        return

    paused = False
    processed_frames = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Достигнут конец видео")
                    break

                # Process frame
                processed_frame, results = trainer.process_frame(frame)

                # Write frame to output video
                out.write(processed_frame)
                processed_frames += 1

                # Display progress
                if processed_frames % 30 == 0:
                    progress = (processed_frames / total_frames) * 100
                    print(f"Обработка: {processed_frames}/{total_frames} кадров ({progress:.1f}%)")

                # Display result
                cv2.imshow('Улучшенный анализ приседаний', processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                trainer.reset_counter()
                print("Счетчик повторений сброшен")
            elif key == ord('p'):
                paused = not paused
                print("Видео приостановлено" if paused else "Видео возобновлено")
            elif key == ord('+'):
                trainer.conf_threshold = min(0.9, trainer.conf_threshold + 0.05)
                print(f"Чувствительность увеличена: {trainer.conf_threshold}")
            elif key == ord('-'):
                trainer.conf_threshold = max(0.1, trainer.conf_threshold - 0.05)
                print(f"Чувствительность уменьшена: {trainer.conf_threshold}")

    except KeyboardInterrupt:
        print("Программа прервана пользователем")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Анализ завершен! Обработано {processed_frames}/{total_frames} кадров")
        print(f"Всего приседаний обнаружено: {trainer.rep_count}")
        print(f"Результат сохранен в: {output_path}")


if __name__ == '__main__':
    main()
