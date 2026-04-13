import subprocess


command_i2i_bg_dream = [
    "accelerate", "launch", "--mixed_precision=bf16", "train_controlnet_i2i_dream.py",
    "--pretrained_model_name_or_path=/stable-diffusion-v1-5",
    "--train_data_dir=data/images",
    "--output_dir=run/sd15-con-i2i-bg-dream",
    "--cache_dir=run/sd15-con-i2i-bg-dream/cache/",
    "--resolution=512",
    "--train_batch_size=8",
    "--gradient_accumulation_steps=4",
    "--max_train_steps=8000",
    # "--num_train_epochs=50",
    "--checkpointing_steps=50",
    "--learning_rate=8e-4",
    "--lr_warmup_steps=0",
    "--seed=42",
    "--allow_tf32",
    "--resume_from_checkpoint=latest",
    "--validation_image", "240507_Oil_3.png", "black_defocus_1.png",
    "--validation_prompt",
    "a photo of w* with oil defect on white background",
    "a photo of w* with baseline defect on black background",
    "--validation_steps=20",
    "--num_validation_images=2",
    "--report_to=tensorboard"
]


command_dreambooth_lora = [
    "accelerate", "launch", "--mixed_precision=bf16", "train_dreambooth_lora.py",
    "--pretrained_model_name_or_path=stable-diffusion-v1-5",
    "--instance_data_dir=/data/seg_raw",
    "--output_dir=run/sd15-dream_lora",

    "--instance_prompt= an image of w*",

    "--resolution=512",
    "--num_train_epochs=50",
    "--train_batch_size=8",
    "--gradient_accumulation_steps=1",
    "--checkpointing_steps=100",
    "--learning_rate=1e-4",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",

    "--validation_prompt=an image of w*",
    "--validation_epochs=100",
    "--num_validation_images=4",

    "--seed=42",
    "--allow_tf32",
    "--report_to=tensorboard",
    "--center_crop",

    "--train_text_encoder",

]


subprocess.run(command_dreambooth_lora)


