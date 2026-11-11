# Image Captioning with CLIP and GPT-2

**IE 7615 Deep Learning for AI · Generative Project · Group 3**
Jongmin Lee, Heewon Myeong

This project builds an image captioning model that writes a sentence for an unseen photo. A frozen CLIP image encoder turns each image into a 512-dimensional embedding, a small mapping network converts it into a prefix of GPT-2 input vectors, and a pretrained GPT-2 decoder continues the prefix as a caption (prefix conditioning, following ClipCap).

```
image ──► CLIP ViT-B/32 (frozen) ──► 512-d ──► mapping network (trained) ──► 10 × 768 prefix ──► GPT-2 ──► caption
caption ──► GPT-2 tokenizer ──► token ids (decoder input and training target)
```

## Milestone status

| Milestone | Scope | Status |
| --- | --- | --- |
| 1 | Dataset preparation, environment setup, image → embedding pipeline, GPT-2 tokenization | Done |
| 2 | Mapping network training, baseline caption generation, decoding strategies (greedy, beam, nucleus) | Planned |

## Repository structure

```
├── Generative_Milestone1_Group3.ipynb   # Milestone 1 notebook (Part A + Part B)
├── data/
│   ├── coco_captions_clean.csv          # 5,000 cleaned image–caption pairs with split labels
│   ├── data_summary.json                # preprocessing config and counts
│   └── clip_image_embs.pt               # CLIP image embeddings (1,000 × 512) with image ids
└── README.md
```

COCO images are not stored in this repository. The notebook downloads them.

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

`coco_captions_clean.csv` columns: `image_id`, `file_name`, `caption`, `split`, plus `ann_id`, `caption_raw`, `n_words`, `width`, `height`, `is_gray`.

## How to run

1. Open `Generative_Milestone1_Group3.ipynb` in Google Colab and set the runtime to **GPU** (T4 is enough).
2. Set `DRIVE_OUT_DIR` in the Part A configuration cell to the folder where outputs should be saved. Part B saves `clip_image_embs.pt` to `save_dir`, which should point to the same folder.
3. Run all cells in order. **Part A must run before Part B**, because Part B uses the Part A table (`sub`) and image folder (`IMG_DIR`).

| Part | What it does | Output |
| --- | --- | --- |
| A. Dataset preparation | Downloads COCO val2017, pairs, cleans, filters, splits | `coco_captions_clean.csv`, `data_summary.json` |
| B. CLIP embedding and GPT-2 tokenization | Embeds the 1,000 images with CLIP, runs the 5-sample test, tokenizes captions | `clip_image_embs.pt` |

Main settings (Part A, cell 0): `N_IMAGES = 1000`, `CAPTIONS_PER_IMAGE = 5`, `MIN_WORDS, MAX_WORDS = 5, 30`, `SPLIT_RATIOS = (0.8, 0.1, 0.1)`, `SEED = 42`.

### Shared Google Drive folder

`/content/drive/MyDrive/...` points to the Drive of whoever runs the notebook. To save into a folder a teammate shared with you, add a shortcut to it in **My Drive** (right-click the folder in *Shared with me* → Organize → Add shortcut → My Drive) and remount Drive with `drive.mount('/content/drive', force_remount=True)`.

## Environment

Tested on Google Colab with an NVIDIA T4 GPU.

| Package | Version |
| --- | --- |
| Python | 3 (Colab default) |
| PyTorch | 2.14.1 (CUDA build) |
| Hugging Face Transformers | 5.22.1 |

Pretrained weights from the Hugging Face Hub: `openai/clip-vit-base-patch32` (CLIP) and `gpt2` (tokenizer). Colab includes all required packages, so no installation cell is needed.

## Milestone 1 results

- 1,000 images embedded into a **1,000 × 512** tensor.
- Sample 5 image–caption test: for each of 5 images, CLIP picked the most similar of 5 captions by cosine similarity and chose the **correct caption 5 of 5 times**. This is a retrieval sanity check, not an evaluation of generated captions.
- All 5,000 captions tokenized with the GPT-2 tokenizer; a 20-token limit covers 99.0% of captions.

## Loading the outputs

```python
import pandas as pd, torch

df = pd.read_csv('data/coco_captions_clean.csv')
emb = torch.load('data/clip_image_embs.pt', weights_only=False)   # {'ids': [...], 'embs': tensor(1000, 512)}

train_ids = set(df.loc[df['split'] == 'train', 'image_id'])
```

`weights_only=False` is needed because the saved image ids are NumPy integers, which recent PyTorch versions do not load in the default safe mode.

## Data license

COCO annotations are released under CC BY 4.0. The images come from Flickr and remain subject to the Flickr terms of use. See [cocodataset.org](https://cocodataset.org).

## Acknowledgments

- ClipCap: R. Mokady, A. Hertz, and A. H. Bermano, "ClipCap: CLIP prefix for image captioning," 2021.
- CLIP: A. Radford et al., "Learning transferable visual models from natural language supervision," ICML 2021.
- GPT-2: A. Radford et al., "Language models are unsupervised multitask learners," OpenAI, 2019.
- COCO: T.-Y. Lin et al., "Microsoft COCO: Common objects in context," ECCV 2014.
