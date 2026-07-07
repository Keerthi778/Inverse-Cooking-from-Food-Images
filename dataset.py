import os
import json
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        A.Rotate(limit=15, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ── Dataset Class ─────────────────────────────────────────────────────────────

class FoodDataset(Dataset):
    def __init__(self, recipes, img_dir, vocab, transform=None):
        self.recipes = recipes
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.num_classes = len(vocab)

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        import torch
        recipe = self.recipes[idx]
        img_id = recipe['id']

        img_path = os.path.join(
            self.img_dir,
            img_id[0], img_id[1], img_id[2], img_id[3],
            img_id + '.jpg'
        )

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        label = torch.zeros(self.num_classes)
        for ingr in recipe.get('ingredients', []):
            ingr_text = ingr['text'].lower().strip()
            if ingr_text in self.vocab:
                label[self.vocab[ingr_text]] = 1.0

        return image, label


# ── Vocabulary Builder ────────────────────────────────────────────────────────

def build_vocab(recipes, min_freq=5):
    from collections import Counter
    counter = Counter()
    for r in recipes:
        for ingr in r.get('ingredients', []):
            counter[ingr['text'].lower().strip()] += 1

    vocab = {ingr: idx for idx, (ingr, cnt)
             in enumerate(counter.most_common()) if cnt >= min_freq}
    inv_vocab = [ingr for ingr, _ in sorted(vocab.items(), key=lambda x: x[1])]
    return vocab, inv_vocab


# ── Recipe Extractor ──────────────────────────────────────────────────────────

class RecipeExtractor:
    BASE_SERVINGS = 4

    def __init__(self, people: int):
        self.people = people
        self.scale = people / self.BASE_SERVINGS

    def scale_ingredients(self, ingredients: list) -> list:
        scaled = []
        for ingr in ingredients:
            qty = ingr.get('qty', 1.0)
            scaled.append({
                'name': ingr['name'],
                'qty': round(qty * self.scale, 2),
                'unit': ingr.get('unit', '')
            })
        return scaled

    def estimate_nutrition(self, ingredient_names: list) -> dict:
        base = {
            'calories': 400,
            'protein_g': 12,
            'carbs_g': 55,
            'fat_g': 14,
            'fiber_g': 4,
        }
        return {k: round(v * self.scale) for k, v in base.items()}

    def classify_cooking_style(self, instructions_text: str) -> str:
        text = instructions_text.lower()
        keywords = {
            'bake':  ['oven', 'bake', 'roast', 'broil'],
            'fry':   ['fry', 'saute', 'sauté', 'pan', 'wok'],
            'grill': ['grill', 'bbq', 'barbecue', 'char'],
            'steam': ['steam', 'boil', 'simmer', 'poach'],
        }
        for style, words in keywords.items():
            if any(w in text for w in words):
                return style
        return 'mixed'


# ── Image Utilities ───────────────────────────────────────────────────────────

def preprocess_single_image(img_path: str):
    """
    Load and preprocess a single image for inference.
    Returns a tensor of shape (1, 3, 224, 224).
    """
    import torch
    import numpy as np
    from PIL import Image as PILImage

    # Try PIL first (most reliable for all formats)
    try:
        pil_img = PILImage.open(img_path).convert('RGB')
        image = np.array(pil_img)
    except Exception:
        # Fallback to OpenCV
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Manual resize and normalize without albumentations
    pil_img = PILImage.fromarray(image).resize((224, 224))
    image = np.array(pil_img).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # HWC → CHW → add batch dim
    image = image.transpose(2, 0, 1)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return tensor


def extract_edges(img_path: str):
    """Canny edge detection for food plating analysis."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((224, 224), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges