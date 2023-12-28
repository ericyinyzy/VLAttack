# VLAttack on the CLIP model

[[CLIP Paper]](https://arxiv.org/abs/2103.00020) 

In this repository, we test VLAttack through the image classification task on the Street View House Number (SVHN) dataset. 
Specifically, we first add a prediction head on the CLIP vision encoder, and then fully finetune the vision encoder on the SVHN dataset.
We then conduct VLAttack on the fine-tuned model using 5K correctly predicted samples. *Note that since this is a uni-modal image classification task, we only employed BSA method at the image attack level.* 
Instructions are shown below:

## Finetuning
You need firstly download the preprocessed SVHN dataset from [train_224.npz](https://drive.google.com/file/d/1q141n6u1lC0sS7nozXUc119p7cKsgwui/view?usp=drive_link) and [test_224.npz](https://drive.google.com/file/d/1UNJqz4Bm7cAQuGUfZP3CFpIyQ3Jqg1Xh/view?usp=drive_link). We have already re-scale each image into a size of 224x224.
The training data is used for finetuning the CLIP vision encoder, and we perform VLAttack on the testing data.
Then, you can finetune the CLIP vision encoder with a `RN50` or `ViT-B/16` structure on the SVHN dataset. For example, to finetune a `RN50` structure, run:
``` bash
python clip_finetune.py --model RN50 --train_path /path/to/train_224.npz --test_path /path/to/test_224.npz
```
The finetuned model weights will be stored in the `./clip_model` directory.
You can also directly download the finetuned model weights from [FT-SVHN-VITB16](https://drive.google.com/file/d/11TyzfGsaskAum1g52DlsxU74aLFz-n-s/view?usp=drive_link) and [FT-SVHN-VITB16](https://drive.google.com/file/d/1-rHPm-ce7Y1GZNwLdQ96ZZP8k5noIjmJ/view?usp=drive_link) and then put them in to the `./clip_model` directory. 
## Attack
To conduct VLAttack on the CLIP model, use the `python clip_attack.py` command with different `--model` and `--method` options. Below are the combinations you can use:

- **Model Options**:
  - RN50
  - ViT-B/16

- **Method Options**:
  - BSA (ours)
  - [DR](https://arxiv.org/pdf/1911.11616.pdf)
  - [FDA](https://arxiv.org/pdf/1909.04385.pdf)
  - [SSP](https://arxiv.org/pdf/2006.04924.pdf)

- **Command**:
  Replace `MODEL_NAME` and `METHOD_NAME` with your chosen options from above:
  ```bash
  python clip_attack.py --test_path /path/to/test_224.npz --model MODEL_NAME --method METHOD_NAME
  ```



