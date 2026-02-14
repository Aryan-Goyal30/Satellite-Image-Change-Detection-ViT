# Satellite Image Change Detection using Vision Transformer (ViT)

A deep learning–based framework for detecting and quantifying spatial changes between multi-temporal satellite images using a pretrained Vision Transformer (ViT).

---

## Overview

This project implements a transformer-based approach for satellite image change detection. By extracting patch-level embeddings from temporal image pairs and computing feature differences, the system identifies and visualizes environmental and structural changes.

The framework generates both visual and quantitative outputs to support urban and environmental monitoring applications.

---

## Key Features

- Patch-level feature extraction using pretrained Vision Transformer (`vit_base_patch16_224`)
- 14×14 change heatmap generation
- Visual overlay highlighting detected regions
- Rule-based multi-class environmental categorization:
  - Flood (Blue)
  - Vegetation Loss (Green)
  - New Construction (Yellow)
  - Fire / Burned Area (Red)
  - Snow / Ice Change (White)
- Percentage area change estimation using adaptive thresholding

---

## Technology Stack

- Python  
- PyTorch  
- timm (Vision Transformer models)  
- NumPy  
- Matplotlib  
- Pillow (PIL)  

---

## Project Structure

```
Satellite-Image-Change-Detection-ViT/
│
├── images/
│   ├── before.png
│   └── after.png
│
├── main.py
└── README.md
```

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Aryan-Goyal30/Satellite-Image-Change-Detection-ViT.git
cd Satellite-Image-Change-Detection-ViT
```

Or download the repository as a ZIP file and extract it.

---

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install torch torchvision timm matplotlib numpy pillow
```

---

### 4. Add Input Images

Place your satellite image pair inside the `images/` folder with the filenames:

```
before.png
after.png
```

---

### 5. Run the Project

```bash
python main.py
```

The system will generate:

- Change heatmap  
- Visual overlay  
- Multi-class environmental change map  
- Percentage of area changed  

---

## Applications

- Urban expansion monitoring  
- Environmental change detection  
- Disaster impact assessment  
- Land-use transformation analysis  

---

## Author

Aryan Goyal  
B.Tech – Computer Science and Engineering  
Manipal University Jaipur  

---

## License

This project is developed for academic and research purposes.
