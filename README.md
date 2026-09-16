# Deep Learning-Based PCB Defect Detection

A Deep Learning project for detecting Printed Circuit Board (PCB) manufacturing defects using YOLO11n, PyTorch, OpenCV, and real-time webcam inspection.

**Course:** Hope AI Institute
**Week:** 11 – Deep Learning
**Developer:** Lokeshwaran R

## Project Overview

This project uses Deep Learning and computer vision to automatically detect defects in Printed Circuit Boards (PCBs).

The system is trained using the YOLO11n object detection model to identify different types of PCB defects from images.

The project also includes real-time webcam detection using OpenCV, where the system identifies defects and displays a **PASS/FAIL** inspection result.

## Objective

The main objective of this project is to develop an automated PCB quality inspection system that can:

* Detect PCB manufacturing defects
* Identify the type of defect
* Locate defects using bounding boxes
* Display detection confidence
* Perform real-time webcam inspection
* Provide a simple PASS/FAIL quality status

## Technologies Used

* Python
* YOLO11n
* Ultralytics
* PyTorch
* TorchVision
* OpenCV
* NumPy
* Matplotlib
* PyYAML
* Pillow
* Jupyter Notebook
* NVIDIA CUDA GPU

## PCB Defect Classes

The model detects six different PCB defect types:

1. Missing Hole
2. Mouse Bite
3. Open Circuit
4. Short
5. Spur
6. Spurious Copper

## Dataset

The project uses the HRIPCB dataset.

The original dataset contains annotated PCB images with Pascal VOC XML annotations.

The XML annotations were converted into YOLO format for training.

### Dataset Statistics

* Annotated images: 693
* Total bounding boxes: 2,953
* Number of classes: 6
* Training images: 485
* Validation images: 138
* Test images: 70
* Train/Validation/Test split: 70% / 20% / 10%

### Class Distribution

| Defect Class    | Bounding Boxes |
| --------------- | -------------: |
| Missing Hole    |            497 |
| Mouse Bite      |            492 |
| Open Circuit    |            482 |
| Short           |            491 |
| Spur            |            488 |
| Spurious Copper |            503 |
| **Total**       |      **2,953** |

## Data Preparation

The original Pascal VOC XML annotations were converted into YOLO annotation format.

The YOLO dataset structure is:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

The dataset was split using a fixed random seed of 42.

## Model

The project uses **YOLO11n** for object detection.

The pretrained YOLO11n model was fine-tuned using the PCB defect dataset.

### Training Configuration

* Model: YOLO11n
* Image size: 640 × 640
* Epochs: 30
* Batch size: 8
* GPU: NVIDIA GeForce RTX 3050 Laptop GPU
* CUDA acceleration: Enabled

The trained model is saved as:

```text
models/pcb_yolo11n/weights/best.pt
```

## Model Performance

The trained model was evaluated on the 70-image test dataset.

| Metric    | Result |
| --------- | -----: |
| Precision |  0.911 |
| Recall    |  0.727 |
| mAP@50    |  0.817 |
| mAP@50-95 |  0.407 |

These results are from the project's test-set evaluation.

## Class-wise Performance

| Defect Class    | Precision | Recall | mAP@50 | mAP@50-95 |
| --------------- | --------: | -----: | -----: | --------: |
| Missing Hole    |     1.000 |  0.928 |  0.992 |     0.590 |
| Mouse Bite      |     0.946 |  0.680 |  0.822 |     0.358 |
| Open Circuit    |     0.875 |  0.784 |  0.799 |     0.346 |
| Short           |     0.939 |  0.872 |  0.938 |     0.466 |
| Spur            |     0.815 |  0.500 |  0.594 |     0.297 |
| Spurious Copper |     0.890 |  0.598 |  0.759 |     0.382 |

## Detection Workflow

The overall workflow of the project is:

```text
PCB Image
    ↓
Image Preprocessing
    ↓
YOLO11n Model
    ↓
Object Detection
    ↓
Defect Classification
    ↓
Bounding Box + Confidence
    ↓
PASS / FAIL Decision
```

If defects are detected:

```text
Defect Detected → FAIL
```

If no defect is detected:

```text
No Defect Detected → PASS
```

## Real-Time Webcam Detection

The project also supports real-time PCB defect detection using a webcam.

OpenCV captures frames from the webcam and sends them to the trained YOLO11n model.

The system displays:

* Detected defect
* Bounding box
* Confidence score
* Number of detected defects
* PASS/FAIL status

Example workflow:

```text
Webcam
   ↓
OpenCV Frame Capture
   ↓
YOLO11n Prediction
   ↓
Defect Detection
   ↓
Confidence Score
   ↓
PASS / FAIL
```

The real-time webcam detection was successfully tested using the NVIDIA RTX 3050 Laptop GPU.

## Project Structure

```text
PCB-DATASET-master/
│
├── dataset/
│   └── data.yaml
│
├── models/
│   └── pcb_yolo11n/
│       ├── weights/
│       │   └── best.pt
│       ├── results.png
│       ├── confusion_matrix.png
│       ├── BoxF1_curve.png
│       ├── BoxP_curve.png
│       ├── BoxPR_curve.png
│       └── BoxR_curve.png
│
├── notebooks/
│   └── PCB_Defect_Detection_YOLO.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/LokeshDataScienceAI/Deep-Learning-Based-PCB-Defect-Detection.git
```

Move into the project directory:

```bash
cd Deep-Learning-Based-PCB-Defect-Detection
```

Create and activate the Python environment:

```bash
conda create -n pcb_yolo python=3.13
conda activate pcb_yolo
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the Jupyter Notebook:

```text
notebooks/PCB_Defect_Detection_YOLO.ipynb
```

Select the `Python (pcb_yolo)` kernel and run the required notebook cells.

The trained model can be loaded using:

```python
from ultralytics import YOLO

model = YOLO("models/pcb_yolo11n/weights/best.pt")
```

## Real-Time Webcam

The trained YOLO model can also be used with an OpenCV webcam stream for real-time PCB inspection.

The webcam system displays the detected PCB defects and the corresponding PASS/FAIL status.

Press **Q** to exit the webcam window.

## Key Features

* Deep Learning-based PCB defect detection
* YOLO11n object detection
* Six PCB defect classes
* Pascal VOC to YOLO annotation conversion
* Train/validation/test dataset split
* Bounding box detection
* Confidence score calculation
* Test-set evaluation
* GPU acceleration using CUDA
* OpenCV real-time webcam detection
* Automatic PASS/FAIL inspection

## Future Enhancements

Possible future improvements include:

* Improve detection performance with additional training data
* Fine-tune YOLO hyperparameters
* Experiment with larger YOLO models
* Add more PCB defect categories
* Improve detection of small defects
* Develop a Flask-based web interface
* Add live camera inspection through a web application
* Deploy the application to a suitable cloud or edge environment
* Add automated inspection reports

## Conclusion

This project demonstrates how Deep Learning and computer vision can be applied to PCB manufacturing quality inspection.

YOLO11n was trained to detect six different PCB defects and was successfully tested on an independent test set and through real-time webcam detection.

The system provides visual defect detection with bounding boxes, confidence scores, and a simple PASS/FAIL inspection result.

## Author

**Lokeshwaran R**

Data Science & AI Learner

GitHub: **LokeshDataScienceAI**

---

**Deep Learning Project | Hope AI Institute | Week 11**
