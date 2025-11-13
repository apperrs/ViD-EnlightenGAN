#     ViD-EnlightenGAN

## Overview
We propose ViD-EnlightenGAN, an unsupervised learning model for low-light video enhancement.

**Key Features:**
- Temporal Omni-Dynamic Inverted Attention Module (TODIAM) for generating an adaptive attention map
- Frame-Stable Attention Module (FSAM) for enhancing inter-frame consistency
- Global-Local Temporal Discriminators with brightness constraints
- Superior performance on SDSD dataset (PSNR: 23.711 dB, SSIM: 0.695)

## Dataset
We use the **SDSD Dataset** for  comprehensive low-light video enhancement research.

Official Website: [SDSD Dataset](https://github.com/dvlab-research/SDSD)

We also use the [DRV](https://github.com/cchen156/Seeing-Motion-in-the-Dark?tab=readme-ov-file) and [Loli-Phone](https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning-Era-Open) datasets to compare the generalization study of real videos.

## Environment Requirements
To run this project, you'll need the following Python packages: 
```bash
torch torchvision torchaudio \
dominate visdom
```
Set up your environment using:
```bash
pip install -r requirements.txt
```

## Training Process
```bash
python3 scripts/script.py --train
```

## Prediction
```bash
python3 scripts/script.py --predict
```

## Key Contributions
- **Temporal Omni-Dynamic Inverted Attention Module (TODIAM):** Replaces original linear attention transformation, incorporating multi-scale spatial feature extraction and Omni-Dimensional Dynamic Convolution (ODConv) for precise focus on key regions with insufficient brightness
- **Frame-Stable Attention Module (FSAM):** Adaptively calibrates feature weights using channel attention mechanism, fusing brightness structure and color information from adjacent frames to enhance inter-frame consistency
- **Global-Local Temporal Discriminators:** The discriminators from EnlightenGAN are adapted into a temporal framework to effectively leverage information from consecutive frames.
- **Global-Local Temporal Brightness Discriminators:** An additional discriminator is proposed to constrain both the overall brightness distribution and local lighting conditions, thereby facilitating natural brightness transitions and preserving details
- **Unsupervised Learning Framework:** Based on EnlightenGAN, effectively handles dynamic low-light scenes without requiring paired training data

## 📬 Contact

For inquiries, collaborations, or feedback:  
✉️ [Email the maintainer](mailto:apperrs@gmail.com)  
💻 Open to contributions and research collaborations  
