#!/usr/bin/env python3
"""
Reykjavik Webcam Object Detection & Line Crossing Counter
Based on Roboflow Supervision tutorial for counting objects crossing lines
Detects and tracks people, cars, bikes, etc. crossing predefined lines
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Dict, List
import time
from datetime import datetime
import argparse
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import re

# Reykjavik webcam stream URL
STREAM_URL = "https://s17.ipcamlive.com/streams/11olozdtumyu7e6fb/stream.m3u8"

def get_stream_url() -> str:
    """Return the stream URL for the Reykjavik webcam using Selenium to wait for iframe"""
    
    # Setup Chrome options for headless browsing
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = None
    try:
        print("Initializing web driver to scrape stream URL...")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to the page
        driver.get("https://livefromiceland.is/webcams/reykjavikurtjorn/")

        # wait 10 seconds to allow page to load
        time.sleep(10)
        
        # Wait for the iframe to load
        print("Waiting for iframe to load...")
        iframe_wait = WebDriverWait(driver, 20)
        iframe = iframe_wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "iframe"))
        )
        
        # Switch to the iframe
        driver.switch_to.frame(iframe)
        
        # Wait for the video element to load within the iframe
        print("Waiting for video element in iframe...")
        video_wait = WebDriverWait(driver, 15)
        video_element = video_wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "video, #mediaplaybackdiv_live_ipc_ic_videoplayer"))
        )
        
        # Try to get the poster attribute or data-src
        stream_url = None
        if video_element.get_attribute("poster"):
            poster_url = video_element.get_attribute("poster")
            stream_url = poster_url.replace("snapshot.jpg", "stream.m3u8")
            print(f"Found poster URL: {poster_url}")
        elif video_element.get_attribute("data-src"):
            stream_url = video_element.get_attribute("data-src")
            print(f"Found data-src: {stream_url}")
        elif video_element.get_attribute("src"):
            stream_url = video_element.get_attribute("src")
            print(f"Found src: {stream_url}")
        
        # Also check for source elements within video
        if not stream_url:
            source_elements = driver.find_elements(By.CSS_SELECTOR, "source")
            for source in source_elements:
                src = source.get_attribute("src")
                if src and "m3u8" in src:
                    stream_url = src
                    print(f"Found source element with m3u8: {src}")
                    break
        
        # If still no stream URL found, try to extract from page source or scripts
        if not stream_url:
            print("Checking page source for stream URLs...")
            page_source = driver.page_source
            
            # Look for m3u8 URLs in the page source
            m3u8_pattern = r'https?://[^\s"\'<>]+\.m3u8'
            matches = re.findall(m3u8_pattern, page_source)
            
            if matches:
                stream_url = matches[0]  # Take the first match
                print(f"Found m3u8 URL in page source: {stream_url}")
        
        if stream_url:
            print(f"✓ Stream URL successfully extracted: {stream_url}")
            return stream_url
        else:
            print("✗ No stream URL found")
            return STREAM_URL  # Fallback to default
            
    except Exception as e:
        print(f"Error scraping stream URL: {e}")
        print("Using fallback stream URL")
        return STREAM_URL  # Fallback to default
        
    finally:
        if driver:
            driver.quit()

def get_stream_url_simple() -> str:
    """Fallback function using simple requests (original method)"""
    try:
        response = requests.get("https://livefromiceland.is/webcams/reykjavikurtjorn/", timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for any iframe sources that might contain the stream
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                if iframe.get('src'):
                    print(f"Found iframe src: {iframe.get('src')}")
            
            # Look for video elements (though they might not be loaded yet)
            video_tags = soup.find_all('video')
            for video in video_tags:
                if video.get('poster'):
                    poster_url = video.get('poster')
                    stream_url = poster_url.replace("snapshot.jpg", "stream.m3u8")
                    print(f"Found video poster: {poster_url}")
                    return stream_url
            
        return STREAM_URL  # Fallback
        
    except Exception as e:
        print(f"Simple scraping failed: {e}")
        return STREAM_URL  # Fallback



class ReykjavikObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.3, 
                 use_grayscale=True, resize_factor=1.0, process_every_n_frames=1,
                 smoothing_length=10):
        """Initialize the object detector with YOLO model"""
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.use_grayscale = use_grayscale
        self.resize_factor = resize_factor
        self.process_every_n_frames = process_every_n_frames
        self.frame_skip_counter = 0
        
        # Store last detections for skipped frames
        self.last_detections = None
        
        # Initialize tracker
        self.tracker = sv.ByteTrack()
        
        # Initialize smoother for detection smoothing
        self.smoother = sv.DetectionsSmoother(length=smoothing_length)
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        
        # Class names for COCO dataset (what YOLOv8 detects)
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow'
        }
        
        # Classes we want to count (vehicles and people)
        self.target_classes = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
        
        print(f"Model loaded. Optimizations: grayscale={use_grayscale}, resize={resize_factor}, skip_frames={process_every_n_frames-1}, smoothing_length={smoothing_length}")
    
    def setup_line_counters(self, frame_shape):
        """Setup line counters for the Reykjavik scene"""
        height, width = frame_shape[:2]
        
        # Define counting lines (you can adjust these based on the view)
        # Horizontal line across the street for vehicles
        self.line1_start = sv.Point(220, 345)
        self.line1_end = sv.Point(374, 384)
        
        # Vertical line for pedestrians near sidewalk
        self.line2_start = sv.Point(97, 711)
        self.line2_end = sv.Point(243, 633)
        
        # Initialize line counters
        self.line_counter1 = sv.LineZone(
            start=self.line1_start, 
            end=self.line1_end,
            triggering_anchors=[sv.Position.CENTER]
        )
        self.line_counter2 = sv.LineZone(
            start=self.line2_start, 
            end=self.line2_end,
            triggering_anchors=[sv.Position.CENTER]
        )
        
        # Line annotators for visualization with smaller labels
        self.line_annotator1 = sv.LineZoneAnnotator(
            thickness=4,
            text_thickness=1,
            text_scale=0.5
        )
        self.line_annotator2 = sv.LineZoneAnnotator(
            thickness=4,
            text_thickness=1,
            text_scale=0.5,
            color=sv.Color.GREEN
        )
        
        print(f"Line counters setup for frame size: {width}x{height}")
        print(f"Line 1 (vehicles): {self.line1_start} to {self.line1_end}")
        print(f"Line 2 (pedestrians): {self.line2_start} to {self.line2_end}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better performance"""
        processed_frame = frame
        original_shape = frame.shape[:2]
        
        # Resize if specified
        if self.resize_factor != 1.0:
            new_width = int(frame.shape[1] * self.resize_factor)
            new_height = int(frame.shape[0] * self.resize_factor)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))
        
        # Convert to grayscale if specified (YOLO can handle grayscale)
        if self.use_grayscale:
            if len(processed_frame.shape) == 3:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                # Convert back to 3-channel for YOLO (duplicate channels)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        
        return processed_frame, original_shape
    
    def scale_detections(self, detections, original_shape, processed_shape):
        """Scale detections back to original frame size"""
        if self.resize_factor == 1.0:
            return detections
        
        scale_x = original_shape[1] / processed_shape[1]
        scale_y = original_shape[0] / processed_shape[0]
        
        # Scale bounding boxes
        detections.xyxy = detections.xyxy * np.array([scale_x, scale_y, scale_x, scale_y])
        
        return detections
    
    def detect_and_track(self, frame):
        """Run detection and tracking on a frame with optimizations"""
        # Frame skipping optimization
        self.frame_skip_counter += 1
        should_process = (self.frame_skip_counter % self.process_every_n_frames) == 0
        
        if should_process:
            # Preprocess frame for performance
            processed_frame, original_shape = self.preprocess_frame(frame)
            
            # Run YOLO detection on processed frame
            results = self.model(processed_frame, conf=self.confidence_threshold, verbose=False)[0]
            
            # Convert to supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            
            # Scale detections back to original size if needed
            if self.resize_factor != 1.0:
                detections = self.scale_detections(detections, original_shape, processed_frame.shape[:2])
            
            # Filter for target classes only
            detections = detections[np.isin(detections.class_id, self.target_classes)]
            
            # Update tracker
            detections = self.tracker.update_with_detections(detections)
            
            # Apply smoothing to reduce jitter
            detections = self.smoother.update_with_detections(detections)
            
            # Store for skipped frames
            self.last_detections = detections
            
            return detections
        else:
            # Use last detections for skipped frames (with updated tracker state)
            return self.last_detections if self.last_detections is not None else sv.Detections.empty()
    
    def annotate_frame(self, frame, detections):
        """Add all annotations to the frame"""
        # Create labels with class names and confidence
        labels = []
        for class_id, confidence, tracker_id in zip(
            detections.class_id, detections.confidence, detections.tracker_id
        ):
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            label = f"#{tracker_id} {class_name} {confidence:.2f}"
            labels.append(label)
        
        # Annotate with bounding boxes and labels
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        
        # Add tracking traces
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        
        return annotated_frame
    
    def update_line_counters(self, detections):
        """Update line counters with current detections"""
        # Update both line counters
        self.line_counter1.trigger(detections)
        self.line_counter2.trigger(detections)
    
    def annotate_lines(self, frame):
        """Draw line counters on frame"""
        # Annotate line 1 (vehicles)
        frame = self.line_annotator1.annotate(
            frame=frame, 
            line_counter=self.line_counter1
        )
        
        # Annotate line 2 (pedestrians)
        frame = self.line_annotator2.annotate(
            frame=frame, 
            line_counter=self.line_counter2
        )
        
        return frame
    
    def add_info_overlay(self, frame):
        """Add information overlay to frame"""
        # Get counts
        line1_in = self.line_counter1.in_count
        line1_out = self.line_counter1.out_count
        line2_in = self.line_counter2.in_count
        line2_out = self.line_counter2.out_count
        
        # Create info text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_lines = [
            f"Reykjavik Live Cam - {timestamp}",
            f"Vehicles Line: IN={line1_in} OUT={line1_out} TOTAL={line1_in + line1_out}",
            f"Pedestrians Line: IN={line2_in} OUT={line2_out} TOTAL={line2_in + line2_out}",
            f"Active Objects: {len(self.tracker.tracked_objects) if hasattr(self.tracker, 'tracked_objects') else 'N/A'}"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (800, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame


def process_live_stream(confidence=0.3, model_path="yolov8n.pt", save_video=False,
                       use_grayscale=True, resize_factor=1.0, process_every_n_frames=1,
                       display_fps_limit=30, smoothing_length=10):
    """Process live stream with object detection and line counting"""
    
    print("Initializing Reykjavik Object Detector...")
    detector = ReykjavikObjectDetector(
        model_path=model_path, 
        confidence_threshold=confidence,
        use_grayscale=use_grayscale,
        resize_factor=resize_factor,
        process_every_n_frames=process_every_n_frames,
        smoothing_length=smoothing_length
    )
    
    print("Opening webcam stream...")
    cap = cv2.VideoCapture(STREAM_URL)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam stream")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Stream opened: {width}x{height} @ {fps}fps")
    
    # Setup video writer if saving
    video_writer = None
    if save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"reykjavik_detected_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        print(f"Saving video to: {output_file}")
    
    # Setup line counters based on first frame
    ret, frame = cap.read()
    if ret:
        detector.setup_line_counters(frame.shape)
    else:
        print("Error: Could not read first frame")
        return
    
    print("\nStarting detection...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'r' - Reset counters")
    print("  'SPACE' - Pause/Resume")
    print("  'f' - Toggle FPS display limit")
    
    frame_count = 0
    screenshot_count = 0
    paused = False
    last_fps_time = time.time()
    current_fps = 0
    processing_fps = 0
    last_display_time = 0
    display_interval = 1.0 / display_fps_limit if display_fps_limit > 0 else 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, attempting to reconnect...")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                process_start = time.time()
                
                # Run detection and tracking
                detections = detector.detect_and_track(frame)
                
                # Update line counters
                detector.update_line_counters(detections)
                
                process_time = time.time() - process_start
                
                # Calculate processing FPS
                if frame_count % 30 == 0:
                    current_time = time.time()
                    current_fps = 30 / (current_time - last_fps_time)
                    processing_fps = 1.0 / process_time if process_time > 0 else 0
                    last_fps_time = current_time
                
                # Limit display FPS to reduce CPU usage
                current_time = time.time()
                if display_interval == 0 or (current_time - last_display_time) >= display_interval:
                    # Annotate frame (only when displaying)
                    annotated_frame = detector.annotate_frame(frame, detections)
                    annotated_frame = detector.annotate_lines(annotated_frame)
                    annotated_frame = detector.add_info_overlay(annotated_frame)
                    
                    # Add performance info
                    perf_text = f"Display FPS: {current_fps:.1f} | Processing FPS: {processing_fps:.1f}"
                    cv2.putText(annotated_frame, perf_text, (width - 400, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Save video frame if enabled
                    if video_writer:
                        video_writer.write(annotated_frame)
                    
                    # Display frame
                    cv2.imshow('Reykjavik Object Detection', annotated_frame)
                    last_display_time = current_time
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reykjavik_screenshot_{timestamp}_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, annotated_frame if not paused else frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset counters
                detector.line_counter1.reset()
                detector.line_counter2.reset()
                print("Counters reset!")
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('f'):
                display_fps_limit = 15 if display_fps_limit == 30 else 30
                display_interval = 1.0 / display_fps_limit
                print(f"Display FPS limit: {display_fps_limit}")
    
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"Frames processed: {frame_count}")
        print(f"Line 1 (Vehicles): IN={detector.line_counter1.in_count}, OUT={detector.line_counter1.out_count}")
        print(f"Line 2 (Pedestrians): IN={detector.line_counter2.in_count}, OUT={detector.line_counter2.out_count}")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Reykjavik Webcam Object Detection")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold (0.0-1.0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video to file")
    parser.add_argument("--test-stream", action="store_true", help="Test stream connectivity only")
    parser.add_argument("--no-grayscale", action="store_true", help="Disable grayscale optimization")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor for processing (0.5 = half size)")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every N frames (1=all, 2=every other, etc.)")
    parser.add_argument("--display-fps", type=int, default=30, help="Limit display FPS (0=unlimited)")
    parser.add_argument("--smoothing", type=int, default=10, help="Detection smoothing length (frames)")
    parser.add_argument("--use-selenium", action="store_true", help="Use Selenium for dynamic stream URL scraping")
    parser.add_argument("--manual-url", type=str, help="Manually specify stream URL")
    
    args = parser.parse_args()
    
    # Handle manual URL override
    global STREAM_URL
    if args.manual_url:
        STREAM_URL = args.manual_url
        print(f"Using manual stream URL: {STREAM_URL}")
    elif args.use_selenium:
        STREAM_URL = get_stream_url()
    else:
        STREAM_URL = get_stream_url_simple()
    
    print("="*60)
    print("Reykjavik Webcam Object Detection & Line Crossing Counter")
    print("="*60)
    print(f"Stream URL: {STREAM_URL}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.confidence}")
    print(f"Save video: {args.save_video}")
    print(f"Grayscale: {not args.no_grayscale}")
    print(f"Resize factor: {args.resize}")
    print(f"Process every N frames: {args.skip_frames}")
    print(f"Display FPS limit: {args.display_fps}")
    print(f"Smoothing length: {args.smoothing}")
    print("="*60)
    
    # Start processing
    process_live_stream(
        confidence=args.confidence,
        model_path=args.model,
        save_video=args.save_video,
        use_grayscale=not args.no_grayscale,
        resize_factor=args.resize,
        process_every_n_frames=args.skip_frames,
        display_fps_limit=args.display_fps,
        smoothing_length=args.smoothing
    )


if __name__ == "__main__":
    # Get the stream URL dynamically (but don't call at module level)
    main()
