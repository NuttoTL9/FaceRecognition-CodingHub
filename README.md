# Face Recognition & Mask Detection System 🎭

A real-time face recognition and mask detection system built with Python and OpenCV.

## Features ✨

- **Real-time Face Recognition**: Identify people from camera feed or video files
- **Mask Detection**: Detect whether a person is wearing a mask or not
- **Multi-face Support**: Process multiple faces simultaneously
- **Easy Integration**: Simple API for integration into other projects

## Demo 🎬

![Demo](demo.gif)

## Installation 🚀

### Prerequisites
- Python 3.7+
- Webcam (for real-time detection)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/NuttoTL9/FaceRecognition-CodingHub.git
cd FaceRecognition-CodingHub
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (if not included):
```bash
# Models will be automatically downloaded on first run
```

## Quick Start 🏃‍♂️

### Basic Usage
```python
python main.py
```

### With custom video file
```python
python main.py --input path/to/video.mp4
```

### With image
```python
python main.py --input path/to/image.jpg
```

## Project Structure 📁

```
FaceRecognition-CodingHub/
├── main.py                 # Main application
├── face_recognition.py     # Face recognition module
├── mask_detection.py       # Mask detection module
├── utils.py               # Utility functions
├── models/                # Pre-trained models
├── data/                  # Training data
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Usage Examples 💡

### Face Recognition
```python
from face_recognition import FaceRecognizer

recognizer = FaceRecognizer()
recognizer.load_known_faces("data/faces/")

# Process image
results = recognizer.recognize_faces(image)
```

### Mask Detection
```python
from mask_detection import MaskDetector

detector = MaskDetector()
mask_results = detector.detect_masks(image)
```

## Configuration ⚙️

Edit `config.py` to customize:
- Detection confidence threshold
- Model paths
- Camera settings
- Output options

## Training Your Own Model 🔧

1. Add training images to `data/faces/` folder
2. Name files as `PersonName_01.jpg`, `PersonName_02.jpg`, etc.
3. Run training script:
```bash
python train.py
```

## Dependencies 📦

```
opencv-python>=4.5.0
numpy>=1.19.0
dlib>=19.22.0
face-recognition>=1.3.0
tensorflow>=2.8.0
matplotlib>=3.3.0
```

## Performance 📊

- **Face Recognition Accuracy**: ~95%
- **Mask Detection Accuracy**: ~98%
- **Processing Speed**: ~30 FPS (with GPU)
- **Memory Usage**: ~500MB

## Troubleshooting 🔧

### Common Issues
1. **Camera not detected**: Check camera permissions
2. **Slow performance**: Consider using GPU acceleration
3. **Import errors**: Ensure all dependencies are installed

### Solutions
```bash
# Reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python

# For macOS dlib issues
brew install cmake
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- OpenCV community
- dlib library developers
- Face recognition library by Adam Geitgey
- TensorFlow team

## Contact 📧

- GitHub: [@NuttoTL9](https://github.com/NuttoTL9)
- Project Link: [https://github.com/NuttoTL9/FaceRecognition-CodingHub](https://github.com/NuttoTL9/FaceRecognition-CodingHub)

## Changelog 📝

### v1.0.0
- Initial release
- Basic face recognition
- Mask detection feature
- Real-time processing

---

⭐ Star this repository if you found it helpful!
