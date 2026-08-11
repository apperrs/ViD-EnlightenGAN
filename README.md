#     ViD-EnlightenGAN: A Temporally Aware GAN for Unsupervised Low-Light Video Enhancement
**[paper](https://ieeexplore.ieee.org/document/11556131)**

## Citation
If you find this project useful in your research, please consider citing:
```bibtex
@ARTICLE{11556131,
  author={Zhang, Heng and Xue, Yijie and Liu, Yanli and Ye, Yiwen and Jiang, Hao and Shu, Feng and Chen, Zhimin},
  journal={IEEE Internet of Things Journal}, 
  title={ViD-EnlightenGAN: A Temporally Aware GAN for Unsupervised Low-Light Video Enhancement}, 
  year={2026},
  volume={13},
  number={16},
  pages={36887-36901},
  keywords={Lighting;Videos;Brightness;Modeling;Visualization;Internet of Things;Convolution;Generators;Sequential analysis;Educational institutions;Attention mechanism;generative adversarial networks;interframe consistency;low-light enhancement;video enhancement},
  doi={10.1109/JIOT.2026.3701729}}
```

## Overview
We propose ViD-EnlightenGAN, an unsupervised learning model for low-light video enhancement.

**Key Features:**
- Temporal Omni-Dynamic Inverted Attention Module (TODIAM) for generating an adaptive attention map
- Frame-Stable Attention Module (FSAM) for enhancing inter-frame consistency
- Global-Local Temporal Discriminators with temporal consistency modeling
- Global-Local Temporal Brightness Discriminators with brightness distribution constraints
- Superior performance on SDSD dataset (PSNR: 23.711 dB, SSIM: 0.695)

## Dataset
We use the **[SDSD Dataset](https://github.com/dvlab-research/SDSD)** for  comprehensive low-light video enhancement research.

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

## License
This project is licensed under the [MIT License](https://github.com/apperrs/ViD-EnlightenGAN/blob/master/LICENSE).
