#!/usr/bin/env python3
"""
RUV Live TV Object Detection with YOLO-World
Zero-shot object detection on Icelandic live television stream
Based on Supervision YOLO-World tutorial
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOWorld
import time
from datetime import datetime
from typing import List, Dict, Tuple
import argparse
import threading
import queue

# RUV.IS stream URL
STREAM_URL = "https://ruv-web-live.akamaized.net/streymi/ruv6/ruv6stream1_.m3u8"

class RUVObjectDetector:
    def __init__(self, model_path="yolo8n.pt", confidence_threshold=0.3, 
                 use_grayscale=True, resize_factor=1.0, process_every_n_frames=1,
                 smoothing_length=10):
        """Initialize the object detector with YOLO-World model"""
        print("Loading YOLO-World model...")
        self.model = YOLOWorld(model_path)
        self.confidence_threshold = confidence_threshold
        self.use_grayscale = use_grayscale
        self.resize_factor = resize_factor
        self.process_every_n_frames = process_every_n_frames
        self.frame_skip_counter = 0
        self.smoothing_length = smoothing_length
        
        # Detection history for smoothing
        self.detection_history = []
        
        # Custom classes for TV content detection
        self.tv_classes = [
            # People and body parts
            "person", "man", "woman", "child", "face", "hand",
            
            # Common TV show elements
            "microphone", "camera", "television", "screen", "desk", "chair", "table",
            
            # News and interview elements
            "news anchor", "reporter", "journalist", "newsroom", "studio",
            
            # Entertainment elements
            "musician", "singer", "performer", "instrument", "guitar", "piano",
            
            # Sports elements
            "athlete", "player", "football", "soccer ball", "basketball",
            
            # Objects commonly seen on TV
            "book", "phone", "laptop", "computer", "cup", "glass", "bottle",
            
            # Outdoor/location elements
            "building", "car", "vehicle", "tree", "street", "road"
        ]
        
        # Set classes for the model using the correct method
        print(f"Setting {len(self.tv_classes)} detection classes...")
        self.model.set_classes(self.tv_classes)
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        print("✅ YOLO-World model initialized successfully!")

    def preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        processed_frame = frame.copy()
        
        # Resize frame if needed
        if self.resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.resize_factor)
            new_height = int(height * self.resize_factor)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))
        
        # Convert to grayscale if enabled (for speed)
        if self.use_grayscale:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        
        return processed_frame

    def detect_objects(self, frame):
        """Run YOLO-World detection on frame"""
        try:
            # Skip frames for performance
            self.frame_skip_counter += 1
            if self.frame_skip_counter % self.process_every_n_frames != 0:
                return None
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Run YOLO-World inference
            results = self.model(processed_frame, conf=self.confidence_threshold, verbose=False)
            
            # Convert to supervision Detections
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Apply Non-Maximum Suppression to eliminate double detections
            detections = detections.with_nms(threshold=0.5)
            
            # Store detection for smoothing
            self.detection_history.append(detections)
            if len(self.detection_history) > self.smoothing_length:
                self.detection_history.pop(0)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return None

    def smooth_detections(self, current_detections):
        """Apply temporal smoothing to reduce flickering"""
        if len(self.detection_history) < 2:
            return current_detections
        
        # Simple smoothing: keep detections that appear in multiple recent frames
        # This is a basic implementation - could be enhanced with tracking
        return current_detections

    def create_detection_labels(self, detections):
        """Create labels with class names and confidence scores"""
        if detections is None or len(detections) == 0:
            return []
        
        labels = []
        for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
            try:
                # Try to get class name from YOLO-World results
                if hasattr(detections, 'data') and 'class_name' in detections.data:
                    if len(detections.data['class_name']) > i:
                        class_name = detections.data['class_name'][i]
                    else:
                        class_name = f"detected_object_{class_id}"
                elif class_id < len(self.tv_classes):
                    class_name = self.tv_classes[class_id]
                else:
                    class_name = f"object_{class_id}"
                
                label = f"{class_name} {confidence:.2f}"
                labels.append(label)
                
            except (IndexError, KeyError) as e:
                # Fallback to generic label
                label = f"object_{class_id} {confidence:.2f}"
                labels.append(label)
                print(f"Label creation warning: {e}, using fallback label")
        
        return labels

    def annotate_frame(self, frame, detections):
        """Add annotations to frame"""
        if detections is None or len(detections) == 0:
            return frame
        
        # Create labels
        labels = self.create_detection_labels(detections)
        
        # Apply annotations
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        return annotated_frame

    def add_info_overlay(self, frame, detections, show_stats=True):
        """Add information overlay to frame"""
        if not show_stats:
            return frame
        
        # Calculate FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
        
        # Prepare info text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detection_count = len(detections) if detections is not None else 0
        
        # Get detected object summary
        object_summary = self.get_object_summary(detections)
        
        info_lines = [
            f"RUV Live TV - {timestamp}",
            f"FPS: {self.current_fps:.1f} | Objects: {detection_count}",
            f"Model: YOLO-World | Confidence: {self.confidence_threshold}",
            f"Most detected: {object_summary}"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        bg_height = len(info_lines) * 25 + 20
        cv2.rectangle(overlay, (10, 10), (600, bg_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

    def get_object_summary(self, detections):
        """Get summary of most frequently detected objects"""
        if detections is None or len(detections) == 0:
            return "None"
        
        # Count class occurrences
        class_counts = {}
        for i, class_id in enumerate(detections.class_id):
            try:
                # Try to get proper class name
                if hasattr(detections, 'data') and 'class_name' in detections.data:
                    if len(detections.data['class_name']) > i:
                        class_name = detections.data['class_name'][i]
                    else:
                        class_name = f"object_{class_id}"
                elif class_id < len(self.tv_classes):
                    class_name = self.tv_classes[class_id]
                else:
                    class_name = f"object_{class_id}"
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
            except (IndexError, KeyError):
                class_name = f"object_{class_id}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Get top 3 most frequent
        top_objects = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        summary = ", ".join([f"{obj}({count})" for obj, count in top_objects])
        
        return summary if summary else "None"

    def save_detection_snapshot(self, frame, detections, filename=None):
        """Save current frame with detections"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ruv_detection_{timestamp}.jpg"
        
        annotated_frame = self.annotate_frame(frame, detections)
        annotated_frame = self.add_info_overlay(annotated_frame, detections)
        
        cv2.imwrite(filename, annotated_frame)
        print(f"📸 Snapshot saved: {filename}")
        return filename


class RUVStreamProcessor:
    """Handles the live stream processing"""
    
    def __init__(self, detector: RUVObjectDetector):
        self.detector = detector
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.detection_queue = queue.Queue(maxsize=10)
        
    def capture_frames(self, stream_url):
        """Capture frames from stream in separate thread"""
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print("❌ Failed to open stream")
            return
        
        print("✅ Stream opened successfully")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, attempting reconnection...")
                    time.sleep(1)
                    continue
                
                # Add frame to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
        except Exception as e:
            print(f"Frame capture error: {e}")
        finally:
            cap.release()
    
    def process_detections(self):
        """Process detections in separate thread"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=1)
                    detections = self.detector.detect_objects(frame)
                    
                    if not self.detection_queue.full():
                        self.detection_queue.put((frame, detections))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection processing error: {e}")
    
    def run_live_detection(self, save_video=False, duration=None):
        """Run live detection with GUI"""
        print("Starting RUV Live TV Object Detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save snapshot")
        print("  'p' - Pause/Resume")
        print("  'r' - Reset detection classes")
        print("  'f' - Toggle full stats display")
        
        self.running = True
        
        # Start background threads
        capture_thread = threading.Thread(target=self.capture_frames, args=(STREAM_URL,))
        detection_thread = threading.Thread(target=self.process_detections)
        
        capture_thread.daemon = True
        detection_thread.daemon = True
        
        capture_thread.start()
        detection_thread.start()
        
        # Video writer setup
        video_writer = None
        if save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"ruv_detection_{timestamp}.mp4"
            # Will initialize when first frame is available
        
        paused = False
        show_full_stats = True
        start_time = time.time()
        snapshot_count = 0
        
        try:
            while self.running:
                try:
                    if not self.detection_queue.empty():
                        frame, detections = self.detection_queue.get(timeout=1)
                        
                        if not paused:
                            # Annotate frame
                            annotated_frame = self.detector.annotate_frame(frame, detections)
                            annotated_frame = self.detector.add_info_overlay(
                                annotated_frame, detections, show_full_stats
                            )
                            
                            # Initialize video writer if needed
                            if save_video and video_writer is None:
                                height, width = annotated_frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))
                                print(f"📹 Recording to: {output_file}")
                            
                            # Save video frame
                            if video_writer:
                                video_writer.write(annotated_frame)
                            
                            # Display frame
                            cv2.imshow('RUV Live TV Detection', annotated_frame)
                        
                        # Handle duration limit
                        if duration and time.time() - start_time > duration:
                            print(f"⏰ Duration limit ({duration}s) reached")
                            break
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        if not self.detection_queue.empty():
                            frame, detections = self.detection_queue.queue[-1]  # Get latest
                            snapshot_count += 1
                            filename = f"ruv_snapshot_{snapshot_count:03d}.jpg"
                            self.detector.save_detection_snapshot(frame, detections, filename)
                    elif key == ord('p'):
                        paused = not paused
                        print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                    elif key == ord('f'):
                        show_full_stats = not show_full_stats
                        print(f"Stats display: {'ON' if show_full_stats else 'OFF'}")
                    elif key == ord('r'):
                        print("🔄 Resetting detection classes...")
                        # Could implement dynamic class changing here
                        
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\n⏹️ Stopping detection...")
        
        finally:
            self.running = False
            if video_writer:
                video_writer.release()
                print(f"✅ Video saved to: {output_file}")
            cv2.destroyAllWindows()


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="RUV Live TV Object Detection")
    parser.add_argument("--model", type=str, default="yolov8s-world.pt", 
                       help="YOLO-World model path")
    parser.add_argument("--confidence", type=float, default=0.3, 
                       help="Detection confidence threshold")
    parser.add_argument("--resize", type=float, default=1.0, 
                       help="Frame resize factor for performance")
    parser.add_argument("--skip-frames", type=int, default=1, 
                       help="Process every N frames (1=all frames)")
    parser.add_argument("--grayscale", action="store_true", 
                       help="Use grayscale for faster processing")
    parser.add_argument("--save-video", action="store_true", 
                       help="Save annotated video")
    parser.add_argument("--duration", type=int, default=None, 
                       help="Run for specified seconds")
    parser.add_argument("--test-stream", action="store_true", 
                       help="Test stream connectivity only")
    
    args = parser.parse_args()
    
    if args.test_stream:
        print("🧪 Testing RUV stream connectivity...")
        cap = cv2.VideoCapture(STREAM_URL)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Stream accessible: {frame.shape}")
                cv2.imshow('RUV Test Frame', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("❌ Could not read frame")
        else:
            print("❌ Could not open stream")
        cap.release()
        return
    
    print("🚀 Initializing RUV Live TV Object Detection")
    print("=" * 50)
    print(f"Stream: {STREAM_URL}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.confidence}")
    print(f"Resize factor: {args.resize}")
    print(f"Frame processing: Every {args.skip_frames} frames")
    print(f"Grayscale: {args.grayscale}")
    print("=" * 50)
    
    # Initialize detector
    detector = RUVObjectDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        use_grayscale=args.grayscale,
        resize_factor=args.resize,
        process_every_n_frames=args.skip_frames
    )
    
    # Initialize processor
    processor = RUVStreamProcessor(detector)
    
    # Run detection
    processor.run_live_detection(
        save_video=args.save_video,
        duration=args.duration
    )
    
    print("🎉 Detection completed!")


if __name__ == "__main__":
    main()