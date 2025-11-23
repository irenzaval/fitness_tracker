import cv2
import numpy as np
from ultralytics import YOLO
import time
import math


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
        self.ideal_knee_angle = 90  # Knee angle at bottom position
        self.ideal_hip_angle = 45  # Hip angle relative to vertical
        self.ideal_back_angle = 20  # Back angle relative to vertical

        # Current state
        self.squat_depth = 0  # Squat depth (0-100%)
        self.feedback_messages = []
        self.rep_count = 0
        self.is_squatting = False
        self.min_knee_angle = 180

        # Colors for feedback
        self.colors = {
            'good': (0, 255, 0),  # Green - good
            'warning': (0, 255, 255),  # Yellow - warning
            'bad': (0, 0, 255),  # Red - bad
            'info': (255, 255, 255)  # White - info
        }

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

    def calculate_vertical_angle(self, a, b):
        """Calculate angle between line and vertical"""
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        # Angle relative to vertical (0-180 degrees)
        angle = np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)
        return angle

    def analyze_squat_form(self, keypoints):
        """Analyze squat form and provide feedback"""
        feedback = []
        warnings = []

        try:
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

            # Calculate angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            knee_angle = (left_knee_angle + right_knee_angle) / 2

            back_angle = self.calculate_vertical_angle(hip_center, shoulder_center)

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
                warnings.append("Too deep!")
            elif knee_angle > 120:
                feedback.append("Go deeper")
            else:
                feedback.append("Good depth")

            if knee_diff > 15:
                warnings.append("Knees not symmetric!")

            # 3. Back analysis
            if back_angle > 40:
                warnings.append("Back too far forward!")
            elif back_angle < 10:
                warnings.append("Too upright")
            else:
                feedback.append("Good back position")

            # 4. Repetition tracking
            if knee_angle < 100 and not self.is_squatting:
                self.is_squatting = True
            elif knee_angle > 160 and self.is_squatting:
                self.is_squatting = False
                self.rep_count += 1
                self.min_knee_angle = 180
                feedback.append(f"Rep {self.rep_count} completed!")

            return feedback, warnings, {
                'knee_angle': knee_angle,
                'back_angle': back_angle,
                'squat_depth': squat_percentage
            }

        except Exception as e:
            return ["Not all body points visible"], [], {}

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
        """Add information panel for squat tracker"""
        h, w = frame.shape[:2]

        # Main info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Info text
        y_offset = 40
        line_height = 25

        cv2.putText(frame, "FITNESS TRAINER: SQUATS", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['info'], 2)
        y_offset += line_height + 10

        cv2.putText(frame, f"Reps: {self.rep_count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info'], 1)
        y_offset += line_height

        cv2.putText(frame, f"Depth: {self.squat_depth:.1f}%", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info'], 1)
        y_offset += line_height

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info'], 1)
        y_offset += line_height

        # Feedback
        cv2.putText(frame, "FEEDBACK:", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info'], 1)
        y_offset += line_height

        for i, message in enumerate(self.feedback_messages[:4]):  # Show up to 4 messages
            color = self.colors['warning'] if "!" in message else self.colors['good']
            cv2.putText(frame, f"- {message}", (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height - 5

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

        cv2.putText(frame, "Depth", (bar_x - 10, bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['info'], 1)

        # Control hints
        cv2.putText(frame, "Controls: Q-quit, R-reset, P-pause", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['info'], 1)

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
            max_det=1,  # Only one person for accurate analysis
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

    # Load video file (replace with your video path)
    video_path = 'squat_video.mp4'  # Change this to your video file path
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video loaded: {video_path}")
    print(f"Video properties: {width}x{height}, {fps:.2f} FPS")
    print("Squat analysis started!")
    print("Instructions:")
    print("- The video will play with real-time squat analysis")
    print("- Press 'P' to pause/resume")
    print("- Press 'R' to reset repetition counter")
    print("- Press 'Q' to quit")

    # Video writer to save output (optional)
    output_path = 'squat_analysis_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break

                # Process frame
                processed_frame, results = trainer.process_frame(frame)

                # Write frame to output video
                out.write(processed_frame)

                # Display result
                cv2.imshow('Fitness Tracker: Squat Analysis', processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                trainer.reset_counter()
                print("Repetition counter reset")
            elif key == ord('p'):
                paused = not paused
                print("Video paused" if paused else "Video resumed")
            elif key == ord('+'):
                trainer.conf_threshold = min(0.9, trainer.conf_threshold + 0.05)
                print(f"Confidence increased: {trainer.conf_threshold}")
            elif key == ord('-'):
                trainer.conf_threshold = max(0.1, trainer.conf_threshold - 0.05)
                print(f"Confidence decreased: {trainer.conf_threshold}")

    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Analysis completed! Total squats detected: {trainer.rep_count}")
        print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()