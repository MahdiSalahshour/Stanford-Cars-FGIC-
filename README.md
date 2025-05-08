```markdown
# Stanford Cars Fine-Grained Classification & Cropping Toolkit

This repository contains two core components for end-to-end fine-grained car model recognition on the [Stanford Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

1. **Cropper Script** (`cropper.py`)  
2. **Classifier Notebook** (`Stanford_Cars_Classifier.ipynb`)  



## 📂 Repository Structure



.
├── cropper.py                   # Stand-alone script to crop bounding-box images
├── Stanford\_Cars\_Classifier.ipynb  # Jupyter notebook for training & evaluation
└── README.md                    # This file

````

---

## 🔧 1. Cropper Script

**File**: `cropper.py`

### Purpose

- **Inspect** the raw dataset directory: print class counts and show a sample image per class.  
- **Crop** every image to its annotated bounding box, saving under `cropped_images/<class_name>/`.

### Usage

```bash
python cropper.py \
  --root-dir /path/to/stanford-car-dataset-by-classes-folder \
  --output-dir /path/to/cropped_images \
  --show-classes 20
````

* `--root-dir` (required): Path to the downloaded Stanford Cars folder (must contain `car_data/train`, `anno_train.csv`, `names.csv`).
* `--output-dir` (default `cropped_images`): Destination for cropped images.
* `--show-classes` (default `20`): Number of classes to preview before cropping.

---

## 🚗 2. Classifier Notebook

**File**: `Stanford_Cars_Classifier.ipynb`

### Overview

Implements a fine-grained car model classifier utilizing:

* **Backbone**: ResNet-50 (ImageNet-pretrained), with only `layer2`, `layer3`, `layer4` unfrozen.
* **Self-Attention**: Spatial attention over the 2048-channel feature map.
* **Bilinear Pooling**: Outer-product pooling on a 512-channel embedding → 512×512 feature vector.
* **MLP Head**: Single linear layer mapping 512² → 196 classes, preceded by signed-sqrt & L2 normalization + dropout.

### Key Sections

1. **Setup & Dependencies**
2. **Data Preparation & Augmentation**
3. **Model Definition**

   * `SelfAttention` module
   * `BilinearCNNWithAttention` model
4. **Training Loop**

   * Loss: `CrossEntropyLoss(label_smoothing=0.1)`
   * Optimizer: Adam with discriminative learning rates
   * Scheduler: `ReduceLROnPlateau`
   * Early stopping
   * (Optional) mixed-precision
5. **Evaluation**

   * Top-1 / Top-5 accuracy
   * Precision, recall, F1 (macro)
   * Confusion matrix visualization

---

## ⚙️ Setup & Installation

Install required Python packages:

```bash
pip install torch torchvision scikit-learn pandas seaborn matplotlib tqdm opencv-python
```

---

## 🗂️ Data Preparation

1. **Run the Cropper** to generate cropped images:

   ```bash
   python cropper.py \
     --root-dir /path/to/stanford-car-dataset-by-classes-folder \
     --output-dir ./cropped_images
   ```

2. **Organize** the cropped output under:

   ```
   cropped_images/
     ├── Class_1_Name/
     │   ├── img001.jpg
     │   └── img002.jpg
     ├── Class_2_Name/
     │   ├── img001.jpg
     │   └── img002.jpg
     └── …
   ```

3. **Set** in the notebook:

   ```python
   data_dir = './cropped_images'
   ```

---

## 🔄 Data Augmentation & Loaders

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
```

* **Split**: 70 % train, 10 % validation, 20 % test
* **Batch size**: 32 (adjust if you hit OOM)
* **Workers**: 4

---

## 🏛️ Model Architecture

### ResNet-50 Backbone

```python
backbone = models.resnet50(pretrained=True)

        # Freeze all layers
        for param in backbone.parameters():
            param.requires_grad = False

        # Make only the last layer (layer4) trainable
        for param in backbone.layer4.parameters():
            param.requires_grad = True
        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer2.parameters():
            param.requires_grad = True        
        # Extract features from the pre-trained ResNet50 backbone, excluding avgpool and fc layers
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.attention = SelfAttention(in_channels=2048)  
        self.reduce_dim = nn.Conv2d(2048, 512, kernel_size=1) 
        self.dropout = nn.Dropout(p=dropout_rate)  
        self.fc = nn.Linear(512 * 512, num_classes)
```

### Self-Attention

```python
class SelfAttention(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        c_q = in_channels // 8
        self.query = nn.Conv2d(in_channels, c_q, 1)
        self.key   = nn.Conv2d(in_channels, c_q, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, x):
        N,C,H,W = x.shape; M = H*W
        q = self.query(x).view(N, C//8, M).permute(0,2,1)
        k = self.key(x).view(N, C//8, M).permute(0,2,1)
        v = self.value(x).view(N,    C, M).permute(0,2,1)
        attn = torch.softmax(q @ k.transpose(-1,-2) / (C//8)**0.5, dim=-1)
        out  = attn @ v
        return out.permute(0,2,1).view(N,C,H,W)
```

### Bilinear Pooling + Classifier

```python
# After attention & 1×1 reduce → (N,512,H,W)
x = x.view(N, 512, -1)                 # (N,512,M)
x = torch.bmm(x, x.transpose(1,2)) / M # (N,512,512)
x = x.view(N, -1)                      # (N,512*512)
x = torch.sign(x) * torch.sqrt(x.abs()+1e-10)
x = F.normalize(x, dim=1)
x = Dropout(x, p=0.6)
logits = Linear(512*512, num_classes)(x)
```

---

## 🎓 Training Configuration

* **Loss**: `CrossEntropyLoss(label_smoothing=0.1)`
* **Optimizer**: Adam (weight\_decay=1e-3)
* **Learning Rates**:

  * Early ResNet layers: 1e-5
  * layer2–4 & attention & reduce\_dim: 1e-4
  * Classifier head: 1e-3
* **Scheduler**: `ReduceLROnPlateau(factor=0.1, patience=2)`
* **EarlyStopping**: patience=5, min\_delta=0.01

---

## 📊 Evaluation Metrics

* **Top-1 / Top-5 Accuracy**
* **Precision / Recall / F1** (macro average)
* **Balanced Accuracy**
* **Confusion Matrix** (first 20 classes)

---

## 📈 Results (Validation)

| Metric         | Value   |
| -------------- | ------- |
| Top-1 Accuracy | 74.07 % |
| Top-5 Accuracy | 92.84 % |
| Precision      | 73.30 % |
| Recall         | 74.73 % |
| F1-Score       | 71.92 % |

---

## 🚀 Next Steps

* Experiment with **EfficientNet** or **ResNeSt** backbones.
* Integrate **multi-head attention** or **Vision Transformers**.
* Add **metric-learning** losses (triplet, ArcFace).
* Deploy the trained model via **TorchServe** or **ONNX**.

---

**Author:** Salahshour
**Date:** 2025-05-08


```
```
