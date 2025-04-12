# 🛵 Helmet Detection System using YOLOv5

![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive computer vision system that detects safety compliance among motorcyclists using YOLOv5. This model can identify multiple classes in motorcycle riding scenarios:

- ✅ **Rider with helmet** - Motorcyclists complying with safety regulations
- ❌ **Rider without helmet** - Motorcyclists violating helmet safety rules
- 🧍 **Rider** - Person riding or sitting on a motorcycle
- 🪖 **Helmet** - Safety helmet object (when not worn)
- 🛵 **Motorcycle with helmet** - Complete motorcycle with compliant rider
- 🚫 **Motorcycle without helmet** - Complete motorcycle with non-compliant rider
- 🔢 **Number plate** - Vehicle registration identification

## 📊 Project Overview

This project leverages the power of YOLOv5 to create a robust detection system for monitoring road safety compliance. The model is trained on a comprehensive dataset of motorcycle riders in various scenarios to accurately identify whether riders are wearing helmets.

## 📁 Dataset Structure

The project uses a structured dataset with the following organization:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Dataset source: [Kaggle - Rider with Helmet/Without Helmet & Number Plate](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate/data)

## 🚀 How It Works

This implementation uses [YOLOv5](https://github.com/ultralytics/yolov5), a state-of-the-art object detection framework. The model processes images or video streams to identify motorcyclists and determine whether they are wearing helmets.

The detection pipeline consists of:
1. Image preprocessing
2. Object detection using the trained YOLOv5 model
3. Classification of detected objects into predefined classes
4. Visualization of results with bounding boxes and labels

## 🛠️ Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/helmet-detection-yolov5.git
   cd helmet-detection-yolov5
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained weights or train your own model.

## 🧠 Model Training

To train the model with your own dataset:

```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt --name helmet_detection
```

Parameters:
- `--img`: Input image size
- `--batch`: Batch size
- `--epochs`: Number of training epochs
- `--data`: Path to data.yaml file
- `--weights`: Initial weights to start from
- `--name`: Name of the training session

## 🔍 Evaluation & Confusion Matrix

The model's performance is evaluated using a confusion matrix to analyze prediction accuracy across all classes.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate and visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
```

## 🖼️ Inference and Visualization

You can run inference on images or video streams:

```python
# For image inference
python detect.py --source path/to/image.jpg --weights path/to/best.pt

# For video inference
python detect.py --source path/to/video.mp4 --weights path/to/best.pt
```

## 📊 Sample Results

The model achieves accurate detection across various scenarios:

![Untitled](https://github.com/user-attachments/assets/1eb24915-813c-4718-84c3-beb412a26c2d)

![Untitled-1](https://github.com/user-attachments/assets/8496bf4d-8783-4bfe-905b-d1c027bb0d0a)

![Untitled](https://github.com/user-attachments/assets/20dffcd5-6391-452d-9276-7f184a330f30)

## 🔄 Deployment

For real-time deployment, you can:

1. Set up the model on an edge device
2. Connect to camera streams
3. Configure alerts for non-compliance
4. Integrate with existing monitoring systems

## ✅ Use Cases

- **Traffic Law Enforcement**: Automated monitoring of helmet law compliance
- **Smart City Infrastructure**: Integration with traffic cameras and management systems
- **Road Safety Analysis**: Collection of helmet usage statistics for policy decisions
- **Corporate Safety Compliance**: Monitoring helmet usage in industrial or corporate environments


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
