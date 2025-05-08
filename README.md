# Fine-Grained Car Model Classification with ResNet-50, Self-Attention & Bilinear Pooling

This notebook implements a high-performance classifier for the [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset (196 classes). It combines:

- **ResNet-50** as a frozen backbone (fine-tuning its top layers)  
- A **Self-Attention** module to let the network focus on key spatial regions  
- **Bilinear pooling** to capture pairwise feature interactions for fine-grained distinctions  
- A lightweight MLP head with **signed-sqrt + L2 normalization** and **label smoothing**

---

## 📁 Contents

1. **Setup & Dependencies**  
2. **Data Preparation & Augmentation**  
3. **Model Architecture**  
   - ResNet-50 feature extractor  
   - Self-Attention block  
   - Bilinear pooling + MLP head  
4. **Training Configuration**  
   - Loss (label smoothing)  
   - Adam optimizer with discriminative learning rates  
   - LR scheduler & early stopping  
   - (Optional) mixed-precision  
5. **Evaluation Metrics**  
   - Top-1 / Top-5 accuracy  
   - Macro-precision, recall, F1  
   - Confusion matrix  
6. **Results & Analysis**  
7. **How to Run**  
8. **Next Steps**

---

## 1. Setup & Dependencies

```bash
pip install torch torchvision scikit-learn pandas seaborn matplotlib tqdm

## 2. Data Preparation & Augmentation
Crop each car image to its bounding box (via supplied CSV annotations).

Organize under:

/data/cropped_images/<class_name>/*.jpg

Train/Val/Test split:

## .70 % train

## .10 % val

## .20 % test
