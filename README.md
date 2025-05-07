# Stanford Cars Fine-Grained Classification with EfficientNet + Self-Attention

This notebook implements a fine-grained car model classifier on the [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset. We combine an EfficientNet‐B0 backbone with a spatial self-attention module and bilinear pooling to capture subtle part-level interactions.  

---

## 📝 Contents

1. **Environment & Dependencies**  
2. **Dataset Preparation**  
3. **Data Augmentation & Loaders**  
4. **Model Architecture**  
   - EfficientNet-B0 feature extractor  
   - Self-Attention module  
   - Bilinear pooling head  
5. **Training Setup**  
   - Loss function with label smoothing  
   - Discriminative learning rates  
   - Scheduler & early stopping  
   - Mixed-precision (optional)  
6. **Evaluation Metrics**  
   - Top-1 / Top-5 accuracy  
   - Precision, recall, F1-score  
   - Confusion matrix  
7. **Results**  
   - Validation & test performance  
   - Sample attention maps  
8. **How to Run**  
9. **Next Steps & Extensions**

---

## 1. Environment & Dependencies

```bash
pip install torch torchvision scikit-learn pandas seaborn matplotlib tqdm


.PyTorch ≥1.8 with CUDA

.torchvision for datasets/transforms

.efficientnet_pytorch for backbone weights

.scikit-learn, pandas, seaborn, matplotlib, tqdm for metrics & visualization

