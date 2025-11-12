# Vision Transformer (ViT) Colab Notebook

**Purpose:** Summarize and demonstrate key concepts from the paper  
*"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)*

---

## Overview

This Colab notebook is designed to explore the Vision Transformer (ViT), a Transformer-based model for image recognition. Unlike traditional convolutional neural networks (CNNs), ViT treats images as sequences of patches, similar to tokens in natural language processing (NLP). This notebook provides a detailed summary of the paper, key insights, and a hands-on demonstration of the ViT model in action.

---

## Sections

### 1. Paper Summary

**Title:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
**Authors:** Alexey Dosovitskiy et al., Google Research, Brain Team  
**Published at:** ICLR 2021  

**Abstract Summary:**  
The Vision Transformer (ViT) applies the standard Transformer architecture directly to sequences of image patches. When trained on large-scale datasets, it achieves state-of-the-art performance on image recognition benchmarks while using fewer computational resources compared to comparable CNNs.

**Key Contributions:**
- Demonstrates that a **pure Transformer architecture** can perform image classification without convolutions.
- Treats **image patches as tokens**, similar to words in NLP.
- Shows that **large-scale pre-training** can compensate for the absence of CNN inductive biases.
- Achieves **superior accuracy** on datasets like ImageNet, CIFAR-100, and VTAB with lower computational cost.

**Model Architecture:**
1. **Patch Embedding:** Images are divided into fixed-size patches (e.g., 16x16) and linearly projected to form embeddings.
2. **[CLS] Token:** A learnable classification token represents the entire image.
3. **Positional Embedding:** Preserves spatial relationships between patches.
4. **Transformer Encoder:** Stack of multi-head self-attention (MSA) and MLP layers.
5. **Classification Head:** Uses the [CLS] token output for prediction.

**Experimental Results:**
- **Datasets:** ImageNet, ImageNet-21k, JFT-300M, CIFAR, VTAB.
- **Models:** ViT-Base (86M params), ViT-Large (307M), ViT-Huge (632M).
- **Performance:** 
  - ImageNet: 88.55% Top-1 accuracy  
  - CIFAR-100: 94.55% accuracy  
  - VTAB: 77.63% accuracy  
- ViT pre-trained on JFT-300M outperformed ResNet-based models using **4× less compute**.

**Insights:**
- CNNs encode spatial priors; ViT learns them from data.  
- ViT requires large-scale pre-training; CNNs may still outperform on smaller datasets.  
- Attention maps show ViT focuses on semantically meaningful regions.  

**Conclusion:**  
ViT establishes a new paradigm for image recognition, prioritizing scale and data over hand-crafted architectural biases. It offers potential for unified architectures across NLP and vision tasks.

---

### 2. Personal Reflections

**Key Learnings:**
1. Vision can be represented as a sequential understanding problem.  
2. Increasing model and dataset size improves performance.  
3. Transformer-based architectures can rival CNNs despite their simplicity.  
4. Large-scale pre-training is crucial for smaller benchmark performance.  

**Future Implications:**  
This approach suggests vision tasks, including segmentation and detection, may eventually be handled by Transformers. Self-supervised and contrastive learning could reduce reliance on labeled data.

**Personal Takeaway:**  
Studying ViT highlights the trade-off between inductive bias and scalability. Future AI architectures may combine the global reasoning of Transformers with CNN efficiency.

---

### 3. Hands-on Demo

The notebook includes a practical demonstration of ViT in action:

1. Loads an example image.  
2. Uses the ViT pretrained model to classify the image.  
3. Visualizes both the input and predicted class.  

```python
!pip install -q torch torchvision transformers pillow

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt

# Load and display image
url = "https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()

# Load pretrained ViT model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Prepare input and run inference
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Predicted label
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]
print(f"Predicted class: {predicted_label}")
plt.imshow(image)
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()
```

---

### 4. Results
**Input**: <img width="266" height="411" alt="download" src="https://github.com/user-attachments/assets/311d30f2-a1c5-41a1-8d7c-a3dd1bfbeaaf" />
**Output**: 
<img width="311" height="435" alt="download" src="https://github.com/user-attachments/assets/78801436-3b41-4da3-b032-4fc3268103e3" />


### 4. Reference

**Paper:** Dosovitskiy, A., et al. *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."* ICLR 2021.  
[Paper Link](https://arxiv.org/abs/2010.11929)


