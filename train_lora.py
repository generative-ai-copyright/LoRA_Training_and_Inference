import os
import subprocess


def train_lora(base_model="runwayml/stable-diffusion-v1-5",
               output_dir=os.getcwd(), dataset=None,
               num_train_epochs=100, checkpointing_steps=1500, max_train_steps=2000,
               learning_rate=1e-04, max_grad_norm=1, mixed_precision="bf16",
               script='text_to_image', instance_prompt=None, validation_prompt=None, train_text_encoder=False):
    command = ""
    if script == 'text_to_image':
        command = [
            'accelerate',
            'launch',
            "--mixed_precision=" + str(mixed_precision),
            'train_text_to_image_lora.py',
            '--pretrained_model_name_or_path=' + str(base_model),
            '--train_data_dir=' + str(dataset),
            '--caption_column=caption',
            '--resolution=512',
            '--random_flip',
            '--train_batch_size=1',
            '--center_crop',
            '--num_train_epochs=' + str(num_train_epochs),
            '--checkpointing_steps=' + str(checkpointing_steps),
            '--max_train_steps=' + str(max_train_steps),
            '--learning_rate=' + str(learning_rate),
            '--max_grad_norm=' + str(max_grad_norm),
            '--lr_scheduler=constant',
            '--lr_warmup_steps=0',
            '--seed=216',
            '--output_dir=' + str(output_dir),
            '--report_to=wandb',
            '--validation_prompt=' + str(validation_prompt)
        ]
    elif script == 'dreambooth':
        if instance_prompt is not None:
            command = [
                'accelerate',
                'launch',
                "--mixed_precision=" + str(mixed_precision),
                'train_dreambooth_lora.py',
                '--pretrained_model_name_or_path=' + str(base_model),
                '--instance_data_dir=' + str(dataset),
                '--instance_prompt=' + str(instance_prompt),
                '--output_dir=' + str(output_dir),
                '--resolution=512',
                '--train_batch_size=1',
                '--center_crop',
                '--num_train_epochs=' + str(num_train_epochs),
                '--checkpointing_steps=' + str(checkpointing_steps),
                '--max_train_steps=' + str(max_train_steps),
                '--learning_rate=' + str(learning_rate),
                '--max_grad_norm=' + str(max_grad_norm),
                '--lr_scheduler=constant',
                '--lr_warmup_steps=0',
                '--seed=216',
                '--report_to=wandb',
                '--validation_prompt=' + str(validation_prompt),
                '--validation_epochs=50',
            ]
            if train_text_encoder:
                command.append('--train_text_encoder')
    else:
        print("Incorrect arguments, please check the script for details.")
        return None
    subprocess.run(command, check=True, stdout=subprocess.PIPE)

    return None
