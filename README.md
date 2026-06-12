# Beyond Spatial Artifacts: A Frequency, Spatial, and Edge Fusion Network for Generalizable Deepfake Detection

## FSFD-Net

FSFD-Net is a multi-domain deepfake detection framework designed to go beyond traditional spatial-only learning by integrating spatial, frequency, and edge-aware representations. The model leverages FFT-based spectral analysis, gradient-based structural cues, and deep convolutional spatial features to improve robustness against advanced face manipulation techniques. Explainability is incorporated through Grad-CAM++ to interpret model decisions and validate learned forensic patterns.

---

## Key Contributions

- Multi-domain feature fusion (Spatial + Frequency + Edge)
- FFT-based frequency-domain forgery analysis
- Edge-aware residual and gradient feature extraction
- EfficientNet-B0 spatial backbone
- Explainable AI using Grad-CAM++
- End-to-end PyTorch implementation
- Robust evaluation using standard classification metrics

---

## Methodology Overview

FSFD-Net consists of three parallel branches:

### Spatial Branch
- EfficientNet-B0 pretrained backbone
- Extracts high-level semantic facial representations

### Frequency Branch
- FFT applied to grayscale input images
- CNN learns spectral inconsistencies caused by manipulation

### Edge Branch
- Gradient-based residual computation
- Captures blending artifacts and boundary inconsistencies

### Feature Fusion
Features from all branches are concatenated and passed through a fully connected classifier for final binary prediction (Real vs Fake).

---

## Dataset Structure

dataset/
 ├── real/
 ├── fake/

Preprocessing:
- Resize images to 224×224
- Convert to tensor
- Normalize pixel values

---

## Training Setup

- Loss Function: CrossEntropyLoss
- Optimizer: Adam (learning rate = 1e-4)
- Early stopping based on validation loss
- Best model saved as best_fsfdnet.pth

---

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Curve

---

## Explainability and Visualization

### Grad-CAM++

Grad-CAM++ is used to visualize regions that contribute most to the model’s prediction. It highlights manipulated facial regions, texture inconsistencies, and blending boundaries, providing interpretability to the classification decision.

Figure: <img width="557" height="341" alt="Gradcam" src="https://github.com/user-attachments/assets/aafa56e7-6cd4-4404-a365-4c239f81ce6d" />


---

### Multi-Domain Feature Analysis

This visualization illustrates how FSFD-Net processes input images across RGB, frequency, and edge domains, demonstrating complementary feature extraction across representations.

Figure: <img width="542" height="339" alt="FSFD-MultiBranch" src="https://github.com/user-attachments/assets/33781bce-c8e9-4385-979e-d3fc783de181" />



---

### Mean FFT Spectrum (Real vs Fake)

This figure compares frequency-domain distributions of real and fake images, highlighting spectral artifacts introduced by manipulation.

Figure: <img width="542" height="339" alt="FSFD-MultiBranch" src="https://github.com/user-attachments/assets/ceab181c-a627-49bd-9650-6cfbbaa39e57" />



---

## Cross-Dataset Generalization Evaluation

To evaluate the robustness and generalization capability of FSFD-Net, cross-dataset testing was performed on a subset of the RVF10K dataset. Unlike in-domain testing, this evaluation measures how well the model adapts to unseen data distributions and manipulation styles.

The model trained on the primary dataset was directly evaluated on RVF10K without fine-tuning.

### Objective
- Assess generalization beyond training distribution
- Evaluate robustness against unseen manipulation artifacts
- Validate frequency-aware feature learning across datasets

### Outcome
FSFD-Net maintains competitive performance on unseen RVF10K samples, indicating that multi-domain fusion (spatial, frequency, and edge features) improves generalization compared to spatial-only baselines with 96% accuracy.


## Results Summary

- Strong performance on binary deepfake classification with 97% accuracy
- Frequency-domain features improve generalization
- Edge features enhance boundary artifact detection
- Grad-CAM++ confirms meaningful model attention regions
- Effective Cross-Data Set evaluation resulting in 96% accuracy

---

## License

This project is intended for research and educational purposes only.
