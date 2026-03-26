# 🖊️ Edge-Optimized Ballpen Defect Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-yellow.svg)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-FP16-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Inference-green.svg)

An edge-optimized computer vision pipeline built to detect manufacturing anomalies in ballpoint pens. Trained on a custom dataset using YOLOv8-nano and compressed via Float16 Quantization to run high-speed, real-time inference on resource-constrained microcontrollers and edge devices.

---

## 📖 Table of Contents
- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [Dataset & Annotation](#dataset--annotation)
- [Training Pipeline](#training-pipeline)
- [Edge Optimization](#edge-optimization)
- [Getting Started (Inference)](#getting-started-inference)
- [Future Work](#future-work)

---

## 🎯 About the Project
High-accuracy AI models are often too heavy to deploy in real-world manufacturing environments where compute resources are limited. This project demonstrates an end-to-end machine learning pipeline that starts with raw image data and ends with a lightweight, deployment-ready model. 

By fine-tuning a state-of-the-art YOLOv8-nano model and converting it to a TensorFlow Lite format, this project achieves high-precision defect detection with a dramatically reduced memory footprint.

## 🛠️ Tech Stack
* **Computer Vision:** YOLOv8-nano (Ultralytics), OpenCV
* **Dataset Management:** Roboflow, Roboflow API
* **Cloud Infrastructure:** Google Colab (NVIDIA T4 GPU)
* **Model Optimization:** TensorFlow Lite (TFLite)

---

## 🗂️ Dataset & Annotation
A highly specific, custom dataset was built from scratch to ensure the model learned the exact features of the manufacturing anomaly.
* **Size:** 100 localized images (50 normal ballpens, 50 ballpens with the specific defect).
* **Annotation:** Manual bounding boxes drawn using **Roboflow**.
* **Splits:** 80% Training / 20% Validation.
* **Format:** Exported in YOLOv8 PyTorch format.

---

## 🚀 Training Pipeline
To maximize training efficiency, the pipeline was executed in the cloud:
1. **Cloud Compute:** Provisioned a Google Colab instance utilizing an **NVIDIA T4 GPU**.
2. **Data Ingestion:** Authenticated via the Roboflow API to securely pull the annotated dataset directly into the notebook.
3. **Transfer Learning:** Fine-tuned the `YOLOv8-nano` baseline model over **50 epochs**. The final layers were successfully re-wired to isolate and detect the unique pen defect. *(Note: Overcame a YAML configuration error during setup by manually bypassing a missing validation split to maintain the rapid prototyping schedule).*

---

## ⚡ Edge Optimization (Quantization)
To ensure the model could run on low-power hardware, the raw PyTorch (`.pt`) weights were compressed.
* **Process:** Applied Float16 (FP16) Quantization.
* **Output:** `best_float16.tflite` (renamed to `defect_model.tflite`).
* **Result:** Drastically reduced file size and optimized inference time (FPS) with negligible loss in detection accuracy.

---

## 💻 Getting Started (Inference)

To test the optimized model locally using OpenCV:

### 1. Clone the repository
```bash
git clone [https://github.com/mjcamunggol/Edge-Hardware-Defect-Detector.git]
cd ballpen-defect-detection
```

### 2. Install Dependencies
pip install opencv-python numpy tensorflow

### 3. Run the inference script

Ensure your camera is connected or provide a test video/image path, then run:

python detect.py --model defect_model.tflite --source 0


## 🔮 Future Work
Deploy the .tflite model directly onto a Raspberry Pi to benchmark real-world inference limits (thermal throttling, FPS drops).

Expand the dataset with various lighting conditions to improve model robustness on the factory floor.

Implement a tracking algorithm (like DeepSORT) to count the total number of defective pens on a moving conveyor belt.