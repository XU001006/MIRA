# **MIRA: Multi-scale Invertible Dual-Attention Redundancy-Aware Network for High-Capacity Video Steganography**

---

## 📌 Overview

Welcome to the official code repository for **MIRA**, a novel deep learning framework designed for **high-capacity video steganography**. MIRA introduces a **multi-scale invertible architecture** with **dual-attention mechanisms** and **redundancy-aware modeling**, enabling the secure and efficient embedding of large payloads into video sequences without compromising visual quality.

This release provides the **core model architecture** to support research reproducibility. While training and evaluation scripts are currently under development and will be released in a future update, this version allows researchers to:

- Reconstruct the MIRA network structure
- Explore the invertible block design
- Integrate components into custom training pipelines


## 🧩 Core Components

The following files constitute the essential modules of the MIRA architecture:

| File | Function |
| ------ |------ |
| `Subnet_constructor.py` | Constructs subnetworks including DBNet and MSFE variants |
| `NLAA.py` | Implements the **Non-Local Attention Architecture (NLAA)** for spatiotemporal feature enhancement |
| `Inv_arch.py` | Defines **invertible blocks (InvBlock)** and **invertible neural networks (InvNN)** for reversible feature transformation |
| `common.py` | Contains shared utility functions and operations (e.g., normalization, reshaping) |
| `networks.py` | Provides model initialization interfaces (e.g., `define_G_v2`) |
| `MIRA.py` | Main implementation of the full MIRA generator network |

---

## 🛠️ How to Use

To instantiate the MIRA model, use the following example:

```
from models.modules.networks import define_G_v2

opt = {
    'model': 'MIRA',
    'num_video': 2,
    'gop': 3,
    'network_G': {
        'which_model_G': {'subnet_type': 'DBNet'},
        'in_nc': 3,
        'out_nc': 3,
        'block_num': [2, 2],
        'block_num_rbm': 8,
        'scale': 4
    }
}

model = define_G_v2(opt)
```

📌 **Requirements:**

- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision
- (Optional) CUDA for GPU acceleration

Install dependencies via:

```
pip install torch torchvision
```

---

## 📚 Citation

Once the full paper is published, we will provide the official citation. In the meantime, please acknowledge this work as:

XU0001006. (2025). *MIRA: Multi-scale Invertible Dual-Attention Redundancy-Aware Network for High-Capacity Video Steganography* [Code]. GitHub.

---

## 📎 License

This project is currently released without a license (`No license` specified). Source code is provided **for academic research purposes only**. All rights reserved by the author. Any commercial use or redistribution is strictly prohibited without prior permission.

We plan to release the full codebase under an open-source license upon publication.

---


*Thank you for your interest in MIRA. We hope this work inspires further innovation in secure multimedia communication.*

