# Assignment 04: Object Detection, Semantic Segmentation, and Face Recognition

**Related to**: Week 6-7 — Object Detection, U-Net, and Face Recognition

## Overview

In this assignment, you will explore advanced CNN applications:
- **Object Detection** using the YOLO library
- **Semantic Segmentation** using U-Net architecture
- **Face Recognition** using DeepFace library
- Understanding **Transpose Convolution** (used in U-Net)

---

## Part 1: Transpose Convolution (Paper Calculation)

Transpose convolution (also called deconvolution or upsampling convolution) is a key operation in architectures like U-Net that need to upsample feature maps back to the original image size.

### Task: Manual Transpose Convolution

Given a **2×2 input** and a **3×3 filter**, compute the transpose convolution output with **stride 2** and **padding 1**.

**Input:**
```
[[2, 1],
 [3, 2]]
```

**Filter (3×3):**
```
[[1, 1, 1],
 [1, 1, 1],
 [1, 1, 1]]
```

**Steps to complete:**

1. Calculate the output size using the formula:
   - Output size = (Input size - 1) × Stride + Filter size - 2 × Padding
   
2. For each input value, place the filter multiplied by that value at the corresponding output location (with stride spacing)

3. Sum overlapping values to get the final output

4. Show the resulting output matrix

**Hint**: Start with a zero-filled output matrix and add each contribution from the input values.

---

## Part 2: Object Detection with YOLO

Use the Ultralytics YOLO library to perform object detection on images.

### Setup

```bash
pip install ultralytics
```

### Tasks

#### 2.1 Basic Object Detection

Load a pre-trained YOLOv8 model and run detection on at least 3 different images:

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # nano model (fast)

# Run inference
results = model('your_image.jpg')
```

**Deliverables:**
- Display the images with bounding boxes drawn
- Print detected objects with their confidence scores
- Show the bounding box coordinates (x_center, y_center, width, height)

#### 2.2 Understanding YOLO Output

For one of your detection results, explain:
- What does the confidence score represent?
- How are bounding box coordinates normalized (0-1 range)?
- What is the difference between `xyxy` format and `xywh` format?

#### 2.3 Non-Max Suppression (NMS)

YOLO uses Non-Max Suppression to filter overlapping boxes.

1. Run detection with different `conf` (confidence) thresholds: 0.25, 0.5, 0.75
2. Run detection with different `iou` thresholds: 0.3, 0.5, 0.7
3. Compare the results and explain what happens when you change these thresholds

```python
results = model('image.jpg', conf=0.5, iou=0.5)
```

#### 2.4 Video Object Detection (Optional Bonus)

Run YOLO on a short video clip and save the output:

```python
results = model('video.mp4', save=True)
```

---

## Part 3: Face Recognition with DeepFace

Use the DeepFace library for face verification and analysis.

### Setup

```bash
pip install deepface
```

### Tasks

#### 3.1 Face Verification

Face verification answers: "Are these two faces the same person?"

```python
from deepface import DeepFace

# Verify if two images are the same person
result = DeepFace.verify(img1_path="face1.jpg", img2_path="face2.jpg")
print(result)
```

**Tasks:**
1. Collect 4-6 face images (2-3 photos of person A, 2-3 photos of person B)
2. Test same-person pairs (should return `verified: True`)
3. Test different-person pairs (should return `verified: False`)
4. Report the distance metrics and explain what they mean

#### 3.2 Face Analysis

Analyze facial attributes:

```python
analysis = DeepFace.analyze(img_path="face.jpg", actions=['age', 'gender', 'emotion'])
print(analysis)
```

**Tasks:**
1. Run analysis on at least 3 different face images
2. Report predicted age, gender, and dominant emotion
3. Discuss accuracy: Were the predictions correct?

#### 3.3 Understanding Face Embeddings

Extract face embeddings (128-dimensional or 512-dimensional vectors):

```python
embedding = DeepFace.represent(img_path="face.jpg", model_name="Facenet")
print(f"Embedding shape: {len(embedding[0]['embedding'])}")
```

**Tasks:**
1. Extract embeddings for 4 face images (2 of person A, 2 of person B)
2. Calculate the Euclidean distance between embeddings:
   - Distance between same-person images
   - Distance between different-person images
3. Visualize: Why is the distance smaller for same-person pairs?

---

## Part 4: U-Net Architecture Understanding

U-Net is a popular architecture for semantic segmentation, featuring an encoder-decoder structure with skip connections.

### 4.1 Architecture Diagram

Draw a diagram (by hand or digitally) of a simplified U-Net architecture with:
- 3 encoder blocks (downsampling path)
- 3 decoder blocks (upsampling path)
- Skip connections between corresponding encoder and decoder blocks

Label the following on your diagram:
- Input image size (e.g., 256×256×3)
- Feature map sizes after each conv/pool/transpose conv operation
- Number of filters at each level (e.g., 64 → 128 → 256 → 128 → 64)

### 4.2 Skip Connections

Answer the following questions:

1. What is the purpose of skip connections in U-Net?
2. How do skip connections help with the "information loss" problem in deep networks?
3. What operation is used to combine encoder features with decoder features? (hint: concatenation)

### 4.3 U-Net Implementation (Simplified)

Implement a basic U-Net building block in PyTorch:

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two consecutive Conv-BatchNorm-ReLU blocks"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Implement two 3x3 convolutions with BatchNorm and ReLU
        pass

    def forward(self, x):
        pass

class DownBlock(nn.Module):
    """Downsampling block: MaxPool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Implement maxpool followed by double convolution
        pass

class UpBlock(nn.Module):
    """Upsampling block: TransposeConv + Concatenate + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Implement transpose convolution for upsampling
        # Then concatenate with skip connection and apply double convolution
        pass
```

**Deliverable**: Complete the implementation and show that shapes are correct by running a test input through each block.

---

## Part 5: Questions for Understanding

Answer the following questions based on the lecture material and your work:

### Object Detection

1. **YOLO vs Sliding Window**: Why is YOLO much faster than traditional sliding window approaches for object detection?

2. **Anchor Boxes**: What problem do anchor boxes solve? How do they help detect multiple objects in the same grid cell?

3. **IoU (Intersection over Union)**: 
   - Write the formula for IoU
   - If IoU ≥ 0.5 is considered a "correct" detection, what does this threshold mean practically?

4. **Bounding Box Regression**: In YOLO, how are bounding box coordinates (bx, by, bw, bh) encoded relative to the grid cell?

### Semantic Segmentation

5. **Classification vs Segmentation**: What is the key difference between image classification, object detection, and semantic segmentation?

6. **Transpose Convolution**: Why do we need transpose convolution in U-Net? What would happen if we only used regular convolutions?

7. **Per-pixel Classification**: In semantic segmentation, we predict a class for every pixel. If the input is 256×256 and we have 10 classes, what is the shape of the output?

### Face Recognition

8. **Verification vs Recognition**: Explain the difference between face verification and face recognition. Which one is a harder problem and why?

9. **One-shot Learning**: Why is face recognition considered a "one-shot learning" problem? How does learning a similarity function solve this?

10. **Triplet Loss**: 
    - What are the three components of triplet loss (Anchor, Positive, Negative)?
    - Why do we need a margin (α) in the triplet loss formula?

11. **Siamese Networks**: How does a Siamese network architecture help with face verification?

---

## Deliverables Summary

Submit a **Jupyter notebook** (`.ipynb`) or **Python script** (`.py`) containing:

1. **Part 1**: Hand-calculated transpose convolution (can be scanned/photo)
2. **Part 2**: YOLO object detection code with visualizations and analysis
3. **Part 3**: DeepFace face verification and analysis code with results
4. **Part 4**: U-Net diagram and PyTorch implementation
5. **Part 5**: Written answers to all questions

---

## Resources

- Ultralytics YOLO: https://docs.ultralytics.com/
- DeepFace: https://github.com/serengil/deepface
- U-Net Paper: https://arxiv.org/abs/1505.04597
- YOLO Paper: https://arxiv.org/abs/1506.02640
- FaceNet Paper: https://arxiv.org/abs/1503.03832
