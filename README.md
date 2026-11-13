# Image Captioning with CLIP and GPT-2

**IE 7615 Deep Learning for AI · Generative Project · Group 3**
Jongmin Lee, Heewon Myeong

This project builds an image captioning model that writes a sentence for an unseen photo. A frozen CLIP image encoder turns each image into a 512-dimensional embedding, a small mapping network converts it into a prefix of GPT-2 input vectors, and a pretrained GPT-2 decoder continues the prefix as a caption (prefix conditioning, following ClipCap).

```
image ──► CLIP ViT-B/32 (frozen) ──► 512-d ──► mapping network (trained) ──► 10 × 768 prefix ──► GPT-2 ──► caption
caption ──► GPT-2 tokenizer ──► token ids (decoder input and training target)
```

## Milestones

| Milestone | Scope | Due | Status |
| --- | --- | --- | --- |
| 1 | Dataset preparation, environment setup, image → embedding pipeline, GPT-2 tokenization | Nov 2026 | Done |
| 2 | CLIP–GPT-2 integration (prefix conditioning), training, baseline captions, decoding strategies | Nov 22 | Done |
| 3 | BLEU, METEOR, CIDEr, ROUGE-L; parameter sensitivity; full fine-tuning vs. LoRA vs. prefix tuning | Nov 28 | Planned |
| 4 | Final report, captions gallery, slides and demo video | Dec 4 | Planned |

## Repository structure

```
├── README.md
├── Generative_Milestone1_Group3.ipynb   # M1: dataset preparation, CLIP embeddings, tokenization
├── Generative_Milestone2_Group3.ipynb   # M2: model, training, caption generation, decoding strategies
├── data/                                # M1 outputs
│   ├── coco_captions_clean.csv          # 5,000 cleaned image–caption pairs with split labels
│   ├── data_summary.json                # preprocessing config and counts
│   └── clip_image_embs.pt               # CLIP image embeddings (1,000 × 512) with image ids
└── results/
    └── milestone2/
        ├── training_log.csv             # train / validation loss per epoch
        ├── training_summary.json        # best epoch, best validation loss, settings
        ├── loss_curve.png
        ├── generated_captions.csv       # 10 test images × 3 decoding strategies
        └── caption_samples.png
```

COCO images are not stored in this repository; the notebooks download them. The trained checkpoint `mapping_net.pt` (126 MB) is above GitHub's 100 MB file limit and is kept in the team's Google Drive folder.

## Dataset

COCO Captions **val2017** (5,000 images, 25,014 captions), used as the full data pool and re-split by image.

| Item | Value |
| --- | --- |
| Images used | 1,000 (random, seed 42) |
| Captions per image | 5 |
| Image–caption pairs | 5,000 |
| Split (images) | train 800 / val 100 / test 100 |
| Caption length | 10.4 words, 11.1 GPT-2 tokens on average (max 34) |

Preprocessing: pair images and captions by `image_id`; lowercase, normalize whitespace and punctuation spacing, drop the trailing period; remove captions shorter than 5 or longer than 30 words and duplicate captions; check every image for corruption; split by `image_id` so no image appears in two splits.

## Model

| Component | Configuration | Parameters | Trained |
| --- | --- | --- | --- |
| CLIP image encoder | ViT-B/32, embeddings precomputed in M1 | — | No |
| Mapping network | MLP 512 → 3,840 → 7,680, reshaped to 10 × 768 | 31,468,800 | Yes |
| GPT-2 decoder | `gpt2` (12 layers, 768 hidden units) | 124,439,808 | No |

The L2-normalized CLIP embedding is mapped to 10 prefix vectors, which are placed in front of the caption token embeddings and fed to GPT-2 as `inputs_embeds`. The loss is cross-entropy on caption tokens with teacher forcing; prefix and padding positions are set to −100.

## How to run

Open the notebooks in Google Colab with a **GPU** runtime (T4 is enough) and run all cells in order.

**Milestone 1** — `Generative_Milestone1_Group3.ipynb`
1. Set `DRIVE_OUT_DIR` in the Part A configuration cell to the output folder. Part B saves `clip_image_embs.pt` to `save_dir`, which should point to the same folder.
2. Part A must run before Part B, because Part B uses the Part A table (`sub`) and image folder (`IMG_DIR`).

**Milestone 2** — `Generative_Milestone2_Group3.ipynb`
1. Set `DATA_SOURCE` in the configuration cell: `'drive'` reads the M1 outputs from `DRIVE_DIR`, `'github'` clones this repository and reads `data/`. Outputs go to `OUT_DIR` (`DRIVE_DIR/milestone2`).
2. Sections 0–8 train the mapping network and save `mapping_net.pt`; sections 9–12 load it and generate captions.
3. If you re-run the model cell, re-run the optimizer cell and the training cell after it. Otherwise the optimizer keeps updating the old model and the loss does not change.

Main settings: prefix length 10, GPT-2 frozen, batch size 32, AdamW (lr 2 × 10⁻⁴, 100 warmup steps, linear decay), up to 20 epochs with early stopping (patience 3), max caption length 30 tokens.

### Shared Google Drive folder

`/content/drive/MyDrive/...` points to the Drive of whoever runs the notebook. To use a folder a teammate shared with you, add a shortcut to it in **My Drive** (right-click the folder in *Shared with me* → Organize → Add shortcut → My Drive) and remount Drive with `drive.mount('/content/drive', force_remount=True)`.

### Loading the outputs

```python
import pandas as pd, torch

df  = pd.read_csv('data/coco_captions_clean.csv')
emb = torch.load('data/clip_image_embs.pt', weights_only=False)   # {'ids': [...], 'embs': tensor(1000, 512)}
model, ckpt = load_model('.../milestone2/mapping_net.pt', device) # load_model is defined in the M2 notebook
```

`weights_only=False` is needed because the saved image ids are NumPy integers, which recent PyTorch versions do not load in the default safe mode.

## Results

### Milestone 1

- 1,000 images embedded into a **1,000 × 512** tensor.
- Sample 5 image–caption test: CLIP picked the correct caption out of 5 for **5 of 5** images (retrieval sanity check).
- All 5,000 captions tokenized with the GPT-2 tokenizer; 4 captions longer than 30 tokens were truncated in M2.

### Milestone 2

| | Value |
| --- | --- |
| Best epoch | 3 of 6 (early stopping) |
| Best validation loss / perplexity | **2.674 / 14.49** |
| Training time | about 30 s per epoch on a T4 |

- Validation loss was lowest at epoch 3 and then rose while training loss kept falling, which indicates overfitting on the 800 training images.
- Captions for 10 test images with greedy decoding, beam search (5 beams), and nucleus sampling (top-p = 0.9): greedy and beam search fell into repetition loops, which `no_repeat_ngram_size = 3` removed. Beam search with this setting is the baseline.
- The model usually captures the main subject and scene but misses actions, small objects, and counts.

## Environment

Tested on Google Colab with an NVIDIA T4 GPU.

| Package | Version |
| --- | --- |
| Python | 3.13 (Colab default) |
| PyTorch | 2.14.1 (CUDA build) |
| Hugging Face Transformers | 5.22.1 |

Pretrained weights from the Hugging Face Hub: `openai/clip-vit-base-patch32` (CLIP) and `gpt2` (GPT-2 and tokenizer). Colab includes all required packages, so no installation cell is needed.

## Data license

COCO annotations are released under CC BY 4.0. The images come from Flickr and remain subject to the Flickr terms of use. See [cocodataset.org](https://cocodataset.org).

## Acknowledgments

- ClipCap: R. Mokady, A. Hertz, and A. H. Bermano, "ClipCap: CLIP prefix for image captioning," 2021.
- CLIP: A. Radford et al., "Learning transferable visual models from natural language supervision," ICML 2021.
- GPT-2: A. Radford et al., "Language models are unsupervised multitask learners," OpenAI, 2019.
- COCO: T.-Y. Lin et al., "Microsoft COCO: Common objects in context," ECCV 2014.
