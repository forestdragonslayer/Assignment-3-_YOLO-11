# Assignment-3-_YOLO-11

# Brain Tumor Detection using YOLOv11

## **Introduction**
This repository contains the implementation of real-time brain tumor detection using the YOLOv11 model. The project includes dataset preparation, model training, inference, and performance evaluation.

## **Installation**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-repo/brain-tumor-yolo11.git
cd brain-tumor-yolo11
```

### **Step 2: Install Dependencies**
Ensure Python (>=3.8) is installed, then install the required packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy matplotlib
```

### **Step 3: Install YOLOv11**
```bash
git clone https://github.com/ultralytics/yolov11.git
cd yolov11
pip install -r requirements.txt
```

## **Dataset Preparation**
1. Download the brain tumor dataset (e.g., from Kaggle or a medical imaging dataset source).
2. Organize the dataset into the following structure:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   ├── labels/
   │   ├── train/
   │   ├── val/
   ```
3. Convert annotations into YOLO format.

## **Training the Model**
Run the following command to train the YOLOv11 model:
```bash
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov11.pt
```

## **Running Inference**
Use the trained model for detection:
```bash
python detect.py --source test_images/ --weights runs/train/exp/weights/best.pt --conf 0.25
```

## **Evaluation**
To evaluate model performance:
```bash
python val.py --data dataset.yaml --weights runs/train/exp/weights/best.pt
```

## **Results**
- Detected tumors will be displayed with bounding boxes.
- The model’s accuracy and performance metrics will be saved in the logs.




