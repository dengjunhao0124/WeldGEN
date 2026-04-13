# WeldGEN: Control-Guided Stable Diffusion for Diverse Weld Image Synthesis

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19557253-blue)](https://doi.org/10.5281/zenodo.19557253)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-green)](https://drive.google.com/drive/folders/1olkdHOuQDYZwWHIHhifn_WLpUeKVYcXv?usp=drive_link)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

Official implementation of **"WeldGen: Control-Guided Stable Diffusion for Diverse Weld Image Synthesis"**, published in *The Visual Computer*.

---

## Requirements

**Hardware**
- CUDA-capable GPU (tested on NVIDIA RTX A5500, 24 GB VRAM)

**Software**
- Python >= 3.10
- PyTorch with CUDA support

Install all dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1olkdHOuQDYZwWHIHhifin_WLpUeKVYcXv?usp=drive_link) and place it under `data/`.

---

The pre-trained Stable Diffusion v1.5 weights can be downloaded from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and placed at `stable-diffusion-v1-5/`.

---

## Usage

### Stage 1 — DreamBooth + LoRA Fine-tuning

Fine-tune Stable Diffusion on the weld segmentation masks to learn the weld concept token `w*`:

```bash
accelerate launch --mixed_precision=bf16 train_dreambooth_lora.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5 \
  --instance_data_dir=data/seg_raw \
  --output_dir=run/sd15-dream_lora \
  --instance_prompt="an image of w*" \
  --resolution=512 \
  --num_train_epochs=50 \
  --train_batch_size=8 \
  --learning_rate=1e-4 \
  --lr_scheduler=constant \
  --train_text_encoder \
  --seed=42 \
  --allow_tf32
```

Or simply run the pre-configured script:

```bash
python run.py
```

The trained LoRA weights will be saved to `run/sd15-dream_lora/`.

### Stage 2 — ControlNet Training (Image-to-Image)

Train ControlNet on top of the DreamBooth-adapted model, using segmentation masks as structural conditioning:

```bash
accelerate launch --mixed_precision=bf16 train_controlnet_i2i_dream.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5 \
  --train_data_dir=data/images \
  --output_dir=run/sd15-con-i2i-bg-dream \
  --resolution=512 \
  --train_batch_size=8 \
  --max_train_steps=8000 \
  --learning_rate=8e-4 \
  --seed=42 \
  --allow_tf32 \
  --report_to=tensorboard
```

The ControlNet weights will be saved to `run/sd15-con-i2i-bg-dream/`.

### Inference

Generate synthetic weld images using the trained model:

```bash
python infer.py
```

Results are saved to `res/`:
```
res/
├── dream_lora/      # Stage 1 generation results
├── i2i-dream/       # Stage 2 ControlNet results
└── mask_to_image/   # Mask-conditioned generation results
```
---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{zhang2025weldgen,
  title     = {WeldGen: Control-Guided Stable Diffusion for Diverse Weld Image Synthesis},
  author    = {Zhang, Qin and Deng, Junhao and Zhao, Zhongyou and Song, Zhelong and
               Wang, Zhenmin and Wang, Hui-ping and Wan, Zixuan and Arinez, Jorge and Li, Guangze},
  journal   = {The Visual Computer},
  year      = {2025},
  doi       = {10.5281/zenodo.19557253}
}
```

---

## Acknowledgements

The training scripts are built upon the [Hugging Face Diffusers](https://github.com/huggingface/diffusers) example scripts for DreamBooth and ControlNet. This work was supported by the National Natural Science Foundation of China (Grant No. U23A20625, U2141216, 52375334) and the Natural Science Foundation of Guangdong Province (Grant No. 2023B1515250003).

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
