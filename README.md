# Traffic Reykjavik - AI-Powered Live Stream Object Detection

A real-time object detection and tracking system for Reykjavik webcams and live TV streams, powered by YOLO models and the Supervision library.

![Demo Screenshot](screenshots/demo_main.jpg)

## 🚀 Features

### Reykjavik Webcam Detection (`main.py`)
- **Real-time object detection** on live Reykjavik webcam streams
- **Line crossing counters** for vehicles and pedestrians
- **Advanced tracking** with ByteTrack for consistent object IDs
- **Detection smoothing** to reduce jitter and false positives
- **Performance optimizations** including grayscale processing, frame skipping, and resizing
- **Dynamic stream URL scraping** from livefromiceland.is
- **Video recording** and screenshot capabilities

### RUV Live TV Detection (`ruv.py`)
- **Zero-shot object detection** using YOLO-World on Icelandic television
- **Custom object classes** optimized for TV content (people, equipment, objects)
- **Multi-threaded processing** for smooth real-time performance
- **Flexible detection targets** for news, entertainment, and sports content

## 📊 Object Detection Capabilities

### Traffic Monitoring (main.py)
- 🚗 **Vehicles**: Cars, trucks, buses, motorcycles
- 🚶 **Pedestrians**: People walking and cycling
- 📈 **Counting**: Bidirectional line crossing detection
- 🎯 **Tracking**: Persistent object IDs across frames

### TV Content Analysis (ruv.py)
- 👥 **People**: Anchors, reporters, guests, performers
- 🎥 **Equipment**: Cameras, microphones, screens, instruments
- 🏢 **Locations**: Studios, outdoor scenes, buildings
- ⚽ **Sports**: Athletes, equipment, sports objects

## 🛠️ Installation

### Prerequisites
```bash
# Create conda environment
conda create -n traffic-detection python=3.10
conda activate traffic-detection

# Install dependencies
pip install ultralytics supervision opencv-python numpy
pip install requests beautifulsoup4 selenium
pip install torch torchvision  # For YOLO models
```

### WebDriver Setup (for dynamic URL scraping)
```bash
# Install ChromeDriver
pip install webdriver-manager
# OR download manually from https://chromedriver.chromium.org/
```

### Clone Repository
```bash
git clone https://github.com/yourusername/traffic-reykjavik.git
cd traffic-reykjavik
```

## 🎮 Usage

### Reykjavik Traffic Detection

#### Basic Usage
```bash
# Run with default settings
python main.py

# Test stream connectivity
python main.py --test-stream

# High performance mode
python main.py --resize 0.5 --skip-frames 2 --display-fps 15
```

#### Advanced Options
```bash
# Custom model and confidence
python main.py --model yolov8s.pt --confidence 0.4

# Enable video recording
python main.py --save-video

# Dynamic URL scraping with Selenium
python main.py --use-selenium

# Manual stream URL
python main.py --manual-url "https://your-stream.m3u8"
```

### RUV Live TV Detection

#### Basic Usage
```bash
# Run TV detection
python ruv.py

# Test RUV stream
python ruv.py --test-stream

# Save video with detections
python ruv.py --save-video --duration 300  # 5 minutes
```

#### Performance Optimization
```bash
# Fast processing mode
python ruv.py --resize 0.7 --skip-frames 3 --grayscale

# Custom confidence threshold
python ruv.py --confidence 0.5
```

## 🎯 Controls

### Keyboard Shortcuts
- **`q`** - Quit application
- **`s`** - Save screenshot with detections
- **`r`** - Reset counting statistics
- **`SPACE`** - Pause/Resume detection
- **`f`** - Toggle FPS display limit
- **`p`** - Pause/Resume (RUV only)

## 📸 Demo Screenshots

### Traffic Detection
![Traffic Overview](screenshots/traffic_overview.jpg)
*Real-time vehicle and pedestrian detection with line counters*

![Line Crossing](screenshots/line_crossing.jpg)
*Bidirectional counting lines for traffic analysis*

![Performance Stats](screenshots/performance_stats.jpg)
*Performance metrics and detection statistics*

### TV Content Detection
![TV Detection](screenshots/ruv_detection.jpg)
*YOLO-World detecting people and objects in live TV*

![News Detection](screenshots/news_detection.jpg)
*News anchor and studio equipment detection*

![Sports Detection](screenshots/sports_detection.jpg)
*Sports content analysis with athlete detection*

## ⚙️ Configuration

### Line Counter Setup
Adjust detection lines in `main.py`:
```python
# Vehicle counting line (horizontal)
self.line1_start = sv.Point(220, 345)
self.line1_end = sv.Point(374, 384)

# Pedestrian counting line (vertical)
self.line2_start = sv.Point(97, 711)
self.line2_end = sv.Point(243, 633)
```

### YOLO-World Classes
Customize TV detection classes in `ruv.py`:
```python
self.tv_classes = [
    "person", "reporter", "anchor",
    "microphone", "camera", "screen",
    "musician", "athlete", "politician"
]
```

## 📈 Performance Optimization

### Speed vs Accuracy Trade-offs

| Setting | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| `--resize 1.0 --skip-frames 1` | Slow | High | Production monitoring |
| `--resize 0.7 --skip-frames 2` | Medium | Good | Live demonstration |
| `--resize 0.5 --skip-frames 3 --grayscale` | Fast | Fair | Resource-constrained |

### Hardware Recommendations
- **CPU**: Modern multi-core processor (Intel i5+ or AMD Ryzen 5+)
- **RAM**: 8GB+ (16GB recommended for simultaneous streams)
- **GPU**: Optional CUDA-compatible GPU for faster inference
- **Network**: Stable internet connection for live streams

## 🔧 Troubleshooting

### Common Issues

#### Stream Connection Problems
```bash
# Test connectivity
python main.py --test-stream
python ruv.py --test-stream

# Try manual URL
python main.py --manual-url "https://backup-stream.m3u8"
```

#### Performance Issues
```bash
# Enable all optimizations
python main.py --resize 0.5 --skip-frames 2 --no-grayscale --display-fps 15

# Reduce detection frequency
python main.py --skip-frames 5 --smoothing 5
```

#### Model Loading Errors
```bash
# Download models manually
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8s-world.pt')"
```

## 📊 Output Files

### Generated Files
- **Screenshots**: `reykjavik_screenshot_YYYYMMDD_HHMMSS_XXX.jpg`
- **Videos**: `reykjavik_detected_YYYYMMDD_HHMMSS.mp4`
- **RUV Snapshots**: `ruv_snapshot_XXX.jpg`
- **RUV Videos**: `ruv_detection_YYYYMMDD_HHMMSS.mp4`

### Statistics Output
```
Final Statistics:
Frames processed: 1503
Line 1 (Vehicles): IN=23, OUT=18
Line 2 (Pedestrians): IN=7, OUT=9
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Ultralytics](https://ultralytics.com/)** for YOLO models
- **[Roboflow Supervision](https://supervision.roboflow.com/)** for computer vision utilities
- **[Live from Iceland](https://livefromiceland.is/)** for webcam streams
- **[RUV](https://ruv.is/)** for live TV streams

## 📞 Support

- Create an [Issue](https://github.com/yourusername/traffic-reykjavik/issues) for bug reports
- Start a [Discussion](https://github.com/yourusername/traffic-reykjavik/discussions) for questions
- Follow the project for updates

---

*Built with ❤️ for the Reykjavik community*