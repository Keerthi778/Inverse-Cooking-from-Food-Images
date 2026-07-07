
## 📌 Overview

**Inverse Cooking** reverses the traditional cooking workflow: instead of searching for a recipe and then buying ingredients, you photograph a dish and the system generates the recipe for you.

Given a food image, the pipeline:
1. Encodes the image into a compact latent representation using a Convolutional Autoencoder
2. Predicts a multi-hot ingredient vector (1,488 possible ingredients)
3. Retrieves the closest matching recipe from 1M+ Recipe1M+ pairs via cosine similarity
4. Scales all quantities to the user's requested serving size
5. Returns the full recipe — title, ingredients, instructions, nutrition info, and cooking style — through a Flask web app

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Convolutional Autoencoder** | Encodes food images into a 256-dim latent space for recipe retrieval |
| 🏷️ **Multi-label Classification** | Predicts 1,488+ ingredient classes from a single image |
| 📏 **Serving Size Scaler** | User inputs people count; all quantities scale proportionally |
| 🔍 **Recipe Retrieval** | Nearest-neighbour cosine similarity over 1M+ Recipe1M+ recipes |
| 🥗 **Nutrition Info** | Per-serving macros (kcal, protein, fat, carbs) via USDA lookup |
| 🍳 **Cooking Style Detection** | Auto-detects bake / grill / fry / steam / roast from instructions |
| 🌐 **Flask Web App** | Full-stack interface with image upload and recipe display |
| 🔐 **User Auth** | SQLite-backed signup, login, session management, and recipe history |

---

## 🗂️ Project Structure

inverse_cooking/

│

├── data/


│   ├── layer1.json          # Recipe1M+ recipe metadata

│   ├── images/              # Recipe1M+ food images (nested layout)

│   └── recipes.pkl          # Preprocessed recipes cache (generated)

│

├── models/

│   ├── __init__.py          # Re-exports all model classes

│   ├── autoencoder.py       # FoodAutoencoder + RecipeHead

│   ├── minivgg.py           # MiniVGG

│   ├── minigooglenet.py     # MiniGoogleNet (Inception modules)

│   ├── minialexnet.py       # MiniAlexNet

│   └── cnn2fc.py            # CNN with 2 Fully Connected Layers

│
├── templates/               # Jinja2 HTML templates

│   ├── base.html

│   ├── index.html           # Upload page

│   ├── result.html          # Recipe output page

│   ├── signup.html

│   └── login.html

│
├── static/

│   ├── uploads/             # User-uploaded images

│   └── style.css

│
├── checkpoints/             # Saved model weights (.pth)

├── dataset.py               # FoodDataset + transforms + vocab builder

├── train.py                 # Training loop (all architectures

├── evaluate.py              # Test-set evaluation + classification report

├── predict.py               # Single-image inference

├── extract.py               # Recipe JSON parsing + nutrition + cooking style

├── app.py                   # Flask application entry point

├── requirements.txt

└── .env                     # SECRET_KEY (not committed)


---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/inverse-cooking.git
cd inverse-cooking
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for significantly faster training.

```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download datasets

**Food-101** (downloaded automatically via torchvision on first run)

**Recipe1M+** — register and download from [im2recipe.csail.mit.edu](http://im2recipe.csail.mit.edu/)

Place the files as follows:
```
data/
├── layer1.json
└── images/          ← nested layout: a/b/c/d/<recipe_id>.jpg
```

### 5. Create a `.env` file

```bash
echo "SECRET_KEY=your_random_secret_key_here" > .env
```

---

## 🚀 Usage

### Train a model

```bash
# Autoencoder (main model — recommended)
python train.py --model autoencoder --epochs 50 --batch_size 32 --latent_dim 256

# Lightweight CNN baseline (fast to test)
python train.py --model cnn2fc --epochs 30 --batch_size 64

# Other architectures
python train.py --model vgg       --epochs 50
python train.py --model googlenet --epochs 40
python train.py --model alexnet   --epochs 40
```

All checkpoints are saved to `checkpoints/<model>_best.pth`.

### Evaluate on the test set

```bash
python evaluate.py --model autoencoder --checkpoint checkpoints/autoencoder_best.pth
python evaluate.py --model vgg         --checkpoint checkpoints/vgg_best.pth
```

### Predict from a single image

```bash
python predict.py \
    --image path/to/food.jpg \
    --checkpoint checkpoints/autoencoder_best.pth \
    --servings 4
```

### Launch the web app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

1. **Sign up** for an account
2. **Upload** any food photo
3. Enter **how many people** to cook for
4. Receive the generated **recipe, ingredients, nutrition info, and cooking instructions**

---

## 🧠 Model Architectures

### Convolutional Autoencoder (`FoodAutoencoder`)

The core model. An encoder compresses the image into a 256-dimensional latent vector; a recipe head classifies ingredients; a decoder reconstructs the original image.

```
Input (3×224×224)
    ↓ Encoder: Conv(stride=2) × 4  →  Flatten  →  FC(256)
    ↓ Latent z  (256,)
    ↓ RecipeHead: FC(512) → FC(256) → FC(1488)   ← ingredient logits
    ↓ Decoder: FC → ConvTranspose × 4 → Sigmoid
Output (3×224×224)
```

**Loss functions:**
- Reconstruction: `MSELoss(recon, input)`
- Ingredient prediction: `BCEWithLogitsLoss(logits, multi_hot_labels)`

### CNN Architectures

| Model | Layers | Params | Notes |
|---|---|---|---|
| **MiniVGG** | 3 × (Conv→BN→ReLU)² + MaxPool + 2×FC | 6.1M | Best accuracy |


| **MiniGoogleNet** | Stem + 4 Inception modules + GAP | 4.3M | Parallel 1×1, 3×3, 5×5 branches |
| **MiniAlexNet** | 5 Conv + 3 FC | 8.7M | Large 11×11 kernel in Conv1 |
| **CNN-2FC** | 2 Conv blocks + 2 FC | 2.1M | Lightest baseline |

All CNN models output raw logits; use `BCEWithLogitsLoss` for multi-label training.

---

## 📊 Results

Evaluated on the Food-101 test set (25,250 images):

| Model | Top-1 Accuracy | Top-5 Accuracy | Params | Epochs |
|---|---|---|---|---|
| **MiniVGG** | **72.4%** | **91.2%** | 6.1M | 50 |
| MiniGoogleNet | 68.9% | 89.7% | 4.3M | 50 |
| MiniAlexNet | 61.3% | 84.1% | 8.7M | 40 |
| CNN-2FC | 54.7% | 79.3% | 2.1M | 30 |

Ingredient prediction (micro-F1 on Recipe1M+ test split):

| Model | Micro-F1 | Precision | Recall |
|---|---|---|---|
| Autoencoder + RecipeHead | **0.61** | 0.74 | 0.52 |
| MiniVGG | 0.57 | 0.71 | 0.47 |

---

## 📦 Dataset Details

| Dataset | Size | Source |
|---|---|---|
| Food-101 | 101,000 images · 101 classes | [Bossard et al., ECCV 2014](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) |
| Recipe1M+ | 1,029,720 recipes · 1M+ image pairs | [Marin et al., TPAMI 2021](http://im2recipe.csail.mit.edu/) |

**Data split (Recipe1M+):**


Total  ──────────────────────────── 1,029,720
         │ 80%            │ 10%   │ 10%
        Train            Val     Test
      ~823,776          ~102,972  ~102,972


---

## 🔐 Authentication

User accounts are stored in a local SQLite database (`users.db`):

- Passwords hashed with **SHA-256** before storage
- Flask **session** management for login state
- Per-user **recipe history** saved with image path and generated recipe JSON

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.x, torchvision |
| Image Processing | OpenCV, Pillow |
| Data / ML Utilities | NumPy, pandas, scikit-learn |
| Web Framework | Flask 3.x |
| Database | SQLite (via Python `sqlite3`) |
| Frontend | HTML5, CSS3, Jinja2 |
| Visualisation | Matplotlib, Seaborn |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Recipe1M+](http://im2recipe.csail.mit.edu/) — Marin et al., TPAMI 2021
- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — Bossard et al., ECCV 2014
- [Inverse Cooking paper](https://arxiv.org/abs/1812.06164) — Salvador et al., CVPR 2019 (original concept inspiration)


---

Made with ❤️ and PyTorch

⭐ Star this repo if you found it useful!
