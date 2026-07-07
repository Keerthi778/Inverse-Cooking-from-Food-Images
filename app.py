"""
app.py — Inverse Cooking Flask Application
"""
import os
import sys
import json
import sqlite3
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as tv_models

from flask import (Flask, render_template, request,
                   redirect, url_for, session, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from dataset import preprocess_single_image, RecipeExtractor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── App Config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'inverse_cooking_secret_key_2024'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXT   = {'png', 'jpg', 'jpeg', 'webp'}
DATABASE      = 'users.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Ingredient Data ───────────────────────────────────────────────────────────
INGREDIENT_QTY = {
    "rice": (300, "g"),        "pasta": (400, "g"),
    "noodles": (300, "g"),     "bread": (4, "slices"),
    "flour": (250, "g"),       "chicken": (500, "g"),
    "beef": (500, "g"),        "lamb": (400, "g"),
    "fish": (400, "g"),        "egg": (3, ""),
    "shrimp": (300, "g"),      "tofu": (350, "g"),
    "paneer": (250, "g"),      "onion": (2, ""),
    "tomato": (3, ""),         "garlic": (4, "cloves"),
    "potato": (3, "medium"),   "carrot": (2, ""),
    "spinach": (200, "g"),     "capsicum": (2, ""),
    "peas": (150, "g"),        "corn": (200, "g"),
    "broccoli": (300, "g"),    "mushroom": (200, "g"),
    "cabbage": (0.5, "head"),  "eggplant": (1, ""),
    "salt": (1, "tsp"),        "pepper": (0.5, "tsp"),
    "cumin": (1, "tsp"),       "turmeric": (0.5, "tsp"),
    "chili": (2, ""),          "coriander": (1, "tsp"),
    "garam masala": (1, "tsp"),"paprika": (1, "tsp"),
    "ginger": (1, "tbsp"),     "cardamom": (4, "pods"),
    "cinnamon": (1, "stick"),  "cloves": (4, ""),
    "butter": (2, "tbsp"),     "cream": (100, "ml"),
    "milk": (200, "ml"),       "cheese": (100, "g"),
    "parmesan": (60, "g"),     "olive oil": (3, "tbsp"),
    "oil": (3, "tbsp"),        "yogurt": (150, "g"),
    "basil": (20, "g"),        "parsley": (15, "g"),
    "cilantro": (20, "g"),     "mint": (15, "g"),
    "thyme": (1, "tsp"),       "oregano": (1, "tsp"),
    "bay leaf": (2, ""),       "soy sauce": (2, "tbsp"),
    "vinegar": (1, "tbsp"),    "lemon": (1, ""),
    "lime": (1, ""),           "tomato sauce": (200, "ml"),
    "coconut milk": (200, "ml"),"sugar": (2, "tbsp"),
    "honey": (2, "tbsp"),      "almonds": (50, "g"),
    "cashews": (50, "g"),      "sesame": (1, "tbsp"),
    "lentils": (300, "g"),     "mustard seeds": (1, "tsp"),
    "cauliflower": (1, "head"),
}

FOOD_PROFILES = {
    "biryani":    ["rice", "chicken", "onion", "tomato", "ginger",
                   "garlic", "garam masala", "turmeric", "mint", "yogurt"],
    "pasta":      ["pasta", "tomato", "garlic", "basil", "olive oil",
                   "parmesan", "onion", "salt", "pepper", "oregano"],
    "curry":      ["onion", "tomato", "garlic", "ginger", "turmeric",
                   "cumin", "coriander", "chili", "oil", "garam masala"],
    "salad":      ["tomato", "capsicum", "onion", "lemon", "olive oil",
                   "salt", "pepper", "parsley", "carrot", "basil"],
    "fried_rice": ["rice", "egg", "onion", "garlic", "soy sauce",
                   "oil", "carrot", "peas", "pepper", "sesame"],
    "soup":       ["onion", "garlic", "carrot", "potato", "tomato",
                   "salt", "pepper", "bay leaf", "thyme", "butter"],
    "stir_fry":   ["chicken", "capsicum", "onion", "garlic", "soy sauce",
                   "oil", "carrot", "broccoli", "ginger", "sesame"],
    "pizza":      ["flour", "tomato sauce", "cheese", "olive oil",
                   "oregano", "garlic", "salt", "basil", "pepper", "onion"],
    "fish_dish":  ["fish", "lemon", "garlic", "butter", "salt",
                   "pepper", "parsley", "olive oil", "thyme", "onion"],
    "dal":        ["lentils", "onion", "tomato", "garlic", "ginger",
                   "turmeric", "cumin", "oil", "salt", "cilantro"],
}

COOKING_STEPS = {
    "biryani": [
        "Wash and soak 300g basmati rice for 30 minutes, then drain.",
        "Marinate chicken with yogurt, turmeric, garam masala and ginger-garlic paste for 20 minutes.",
        "Heat oil in a heavy pot. Fry sliced onions until golden brown.",
        "Add marinated chicken, cook on high 5 minutes then medium for 10 minutes.",
        "Parboil the soaked rice with salt and whole spices until 70% cooked.",
        "Layer parboiled rice over chicken. Add fried onions and mint on top.",
        "Seal pot with foil. Cook on low flame (dum) for 20-25 minutes.",
        "Rest 10 minutes before opening. Mix gently and serve hot with raita.",
    ],
    "pasta": [
        "Boil a large pot of salted water. Cook pasta until al dente (8-10 min).",
        "Heat olive oil in a wide pan. Add minced garlic, saute 60 seconds.",
        "Add diced tomatoes and cook on medium heat for 8-10 minutes.",
        "Season with salt, pepper, oregano and chili flakes.",
        "Add drained pasta to sauce. Toss well, adding pasta water if needed.",
        "Remove from heat. Stir in fresh basil and grated parmesan.",
        "Plate and finish with extra parmesan and a drizzle of olive oil.",
    ],
    "curry": [
        "Heat oil in a deep pan. Add whole spices and let them splutter.",
        "Add finely chopped onions, cook until deep golden brown.",
        "Add ginger-garlic paste, saute 2 minutes until raw smell disappears.",
        "Add tomatoes and cook until oil separates (8-10 minutes).",
        "Add turmeric, cumin, coriander, chili powder and salt. Fry 1 minute.",
        "Add main protein or vegetables, coat well with masala. Add water.",
        "Cover and simmer 15-20 minutes. Finish with garam masala and cilantro.",
    ],
    "salad": [
        "Wash and dry all vegetables thoroughly.",
        "Chop tomatoes, capsicum, carrot and onion into equal bite-sized pieces.",
        "Combine all vegetables in a large mixing bowl.",
        "Make dressing: whisk olive oil, lemon juice, salt and pepper together.",
        "Pour dressing over salad and toss gently to coat everything evenly.",
        "Taste and adjust seasoning. Refrigerate 15 minutes to meld flavours.",
        "Serve chilled, garnished with fresh parsley leaves.",
    ],
    "fried_rice": [
        "Use cold cooked rice (day-old rice works best for frying).",
        "Heat wok or large pan on very high heat. Add oil until it shimmers.",
        "Add minced garlic and ginger, stir-fry for 30 seconds.",
        "Push to sides, scramble eggs in center, then mix everything together.",
        "Add vegetables (carrot, peas) and stir-fry for 3-4 minutes.",
        "Add cold rice, break up clumps, toss on high heat for 3-4 minutes.",
        "Season with soy sauce, sesame oil, salt and pepper. Serve hot.",
    ],
    "soup": [
        "Heat butter in a large pot. Saute onions and garlic until soft.",
        "Add diced vegetables (carrot, potato) and cook for 3-4 minutes.",
        "Pour in 1 litre of stock or water. Add bay leaf, thyme, salt and pepper.",
        "Bring to a boil, then reduce heat. Simmer 20-25 minutes.",
        "Blend half the soup for a creamier texture, then recombine.",
        "Stir in cream for richness. Taste and adjust seasoning.",
        "Serve hot in bowls garnished with fresh parsley.",
    ],
    "stir_fry": [
        "Cut all vegetables and protein into thin uniform slices.",
        "Mix sauce: soy sauce, ginger, garlic, sesame oil and a pinch of sugar.",
        "Heat wok on highest flame until smoking. Add oil.",
        "Add protein first, sear 90 seconds without moving, then stir-fry 2 minutes. Remove.",
        "Add harder vegetables first (broccoli, carrot), stir-fry 2 minutes.",
        "Return protein to wok. Pour sauce over everything, toss for 1 minute.",
        "Finish with sesame seeds. Serve immediately over steamed rice.",
    ],
    "pizza": [
        "Mix flour, yeast, salt and olive oil with warm water. Knead into dough.",
        "Let dough rise in a warm place for 1 hour until doubled in size.",
        "Preheat oven to 220°C (425°F). Roll dough into a thin round base.",
        "Spread tomato sauce evenly over the base, leaving a border.",
        "Top with cheese, vegetables and seasonings of your choice.",
        "Bake for 12-15 minutes until crust is golden and cheese is bubbling.",
        "Slice and serve immediately with fresh basil.",
    ],
    "fish_dish": [
        "Pat fish dry with paper towels. Season with salt, pepper and lemon zest.",
        "Heat butter and olive oil in a pan over medium-high heat.",
        "Place fish skin-side down. Cook without moving for 3-4 minutes.",
        "Flip carefully. Add garlic and thyme to the pan.",
        "Baste fish with the butter. Cook 2-3 more minutes.",
        "Squeeze lemon juice over the fish. Remove from heat.",
        "Serve immediately garnished with fresh parsley.",
    ],
    "dal": [
        "Rinse lentils thoroughly until water runs clear. Soak 20 minutes.",
        "Boil lentils in 3 cups water with turmeric and salt until soft (20 min).",
        "Heat oil in a separate pan. Add mustard seeds and let them pop.",
        "Add onions and cook until golden. Add ginger-garlic paste, cook 2 minutes.",
        "Add tomatoes and cook until mushy. Add cumin and coriander powder.",
        "Pour the tempering over cooked lentils. Stir and simmer 5 minutes.",
        "Finish with fresh cilantro. Serve hot with rice or bread.",
    ],
    "default": [
        "Gather and prepare all ingredients — wash, peel and chop as needed.",
        "Heat your cooking vessel over medium heat and add oil or butter.",
        "Start with aromatics: cook onions until translucent, then add garlic.",
        "Add main ingredients, cooking harder items first.",
        "Season progressively with salt, pepper and spices.",
        "Taste and adjust seasoning. Add fresh herbs at the very end.",
        "Rest the dish 2-3 minutes off heat before plating and serving.",
    ],
}

NUTRITION_BY_TYPE = {
    'biryani':    {'calories': 520, 'protein_g': 22, 'carbs_g': 68, 'fat_g': 18, 'fiber_g': 3},
    'pasta':      {'calories': 420, 'protein_g': 14, 'carbs_g': 62, 'fat_g': 12, 'fiber_g': 4},
    'curry':      {'calories': 380, 'protein_g': 24, 'carbs_g': 22, 'fat_g': 20, 'fiber_g': 5},
    'salad':      {'calories': 180, 'protein_g': 6,  'carbs_g': 18, 'fat_g': 10, 'fiber_g': 6},
    'fried_rice': {'calories': 450, 'protein_g': 16, 'carbs_g': 65, 'fat_g': 14, 'fiber_g': 3},
    'soup':       {'calories': 220, 'protein_g': 10, 'carbs_g': 28, 'fat_g': 8,  'fiber_g': 5},
    'stir_fry':   {'calories': 320, 'protein_g': 28, 'carbs_g': 20, 'fat_g': 14, 'fiber_g': 5},
    'pizza':      {'calories': 500, 'protein_g': 20, 'carbs_g': 58, 'fat_g': 22, 'fiber_g': 3},
    'fish_dish':  {'calories': 280, 'protein_g': 32, 'carbs_g': 8,  'fat_g': 14, 'fiber_g': 2},
    'dal':        {'calories': 280, 'protein_g': 16, 'carbs_g': 42, 'fat_g': 6,  'fiber_g': 10},
    'default':    {'calories': 400, 'protein_g': 18, 'carbs_g': 45, 'fat_g': 14, 'fiber_g': 4},
}

STYLE_BY_TYPE = {
    'biryani': 'steam', 'pasta': 'boil', 'curry': 'fry',
    'salad': 'raw', 'fried_rice': 'fry', 'soup': 'boil',
    'stir_fry': 'fry', 'pizza': 'bake', 'fish_dish': 'grill', 'dal': 'boil',
}

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT    UNIQUE NOT NULL,
        password TEXT    NOT NULL,
        created  TEXT    DEFAULT (datetime('now'))
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL,
        image_path  TEXT,
        ingredients TEXT,
        nutrition   TEXT,
        style       TEXT,
        people      INTEGER,
        created     TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    db.commit()
    db.close()
    print("[DB] Initialised users.db")

# ── Model ─────────────────────────────────────────────────────────────────────
_model_cache = {}

def load_model():
    if 'encoder' in _model_cache:
        return _model_cache
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet  = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    encoder.eval().to(device)
    _model_cache['encoder'] = encoder
    _model_cache['device']  = device
    print("[Model] Loaded pretrained ResNet50")
    return _model_cache

def classify_food(feat_vec, img_path):
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return 'curry'
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r = float(img_rgb[:,:,0].mean())
    g = float(img_rgb[:,:,1].mean())
    b = float(img_rgb[:,:,2].mean())
    h = float(img_hsv[:,:,0].mean())
    s = float(img_hsv[:,:,1].mean())
    v = float(img_hsv[:,:,2].mean())
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if g > r + 15 and g > b + 10:
        return 'salad'
    if 10 < h < 30 and s > 100 and r > 140:
        return 'biryani' if texture > 800 else 'curry'
    if r > g + 20 and r > b + 20:
        return 'pasta' if texture < 400 else 'pizza'
    if v < 110 and s > 80:
        return 'stir_fry' if g > 80 else 'fried_rice'
    if s < 60 and v > 150:
        return 'soup'
    if 20 < h < 35 and s < 120:
        return 'dal'
    if b > r + 10:
        return 'fish_dish'
    fallbacks = ['curry', 'stir_fry', 'soup', 'fried_rice', 'pasta']
    return fallbacks[int(feat_vec.mean() * 1000) % 5]

def predict_recipe(img_path: str, people: int = 4):
    cache  = load_model()
    enc    = cache['encoder']
    device = cache['device']

    tensor = preprocess_single_image(img_path).to(device)

    with torch.no_grad():
        features = enc(tensor)
        feat_vec = features.squeeze().cpu().numpy()

    food_type        = classify_food(feat_vec, img_path)
    ingredient_names = FOOD_PROFILES.get(food_type, FOOD_PROFILES['curry'])[:10]

    extractor  = RecipeExtractor(people)
    ingr_dicts = []
    for name in ingredient_names:
        qty, unit = INGREDIENT_QTY.get(name, (100, 'g'))
        ingr_dicts.append({'name': name, 'qty': float(qty), 'unit': unit})

    scaled    = extractor.scale_ingredients(ingr_dicts)
    nutrition = NUTRITION_BY_TYPE.get(food_type, NUTRITION_BY_TYPE['default']).copy()
    steps     = COOKING_STEPS.get(food_type, COOKING_STEPS['default'])
    style     = STYLE_BY_TYPE.get(food_type, 'mixed')

    return {
        'ingredients': scaled,
        'nutrition':   nutrition,
        'style':       style,
        'steps':       steps,
        'people':      people,
        'dish_type':   food_type.replace('_', ' ').title(),
    }

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('signup.html')
        db = get_db()
        existing = db.execute('SELECT id FROM users WHERE username=?',
                              (username,)).fetchone()
        if existing:
            flash('Username already taken.', 'error')
            db.close()
            return render_template('signup.html')
        db.execute('INSERT INTO users (username, password) VALUES (?,?)',
                   (username, generate_password_hash(password)))
        db.commit()
        db.close()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        db   = get_db()
        user = db.execute('SELECT * FROM users WHERE username=?',
                          (username,)).fetchone()
        db.close()
        if user and check_password_hash(user['password'], password):
            session['user_id']  = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if 'image' not in request.files:
        flash('No image uploaded.', 'error')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Please upload a valid image (jpg, png, webp).', 'error')
        return redirect(url_for('index'))

    people   = max(1, min(20, int(request.form.get('people', 4))))
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    result = predict_recipe(save_path, people)

    db = get_db()
    db.execute(
        '''INSERT INTO predictions
           (user_id, image_path, ingredients, nutrition, style, people)
           VALUES (?,?,?,?,?,?)''',
        (session['user_id'], save_path,
         json.dumps(result['ingredients']),
         json.dumps(result['nutrition']),
         result['style'], people)
    )
    db.commit()
    db.close()

    return render_template('result.html',
                           result=result,
                           image_path=save_path)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    db   = get_db()
    rows = db.execute(
        '''SELECT * FROM predictions WHERE user_id=?
           ORDER BY created DESC LIMIT 20''',
        (session['user_id'],)
    ).fetchall()
    db.close()
    entries = []
    for row in rows:
        entries.append({
            'id':          row['id'],
            'image_path':  row['image_path'],
            'ingredients': json.loads(row['ingredients']),
            'nutrition':   json.loads(row['nutrition']),
            'style':       row['style'],
            'people':      row['people'],
            'created':     row['created'],
        })
    return render_template('history.html', entries=entries)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='127.0.0.1', port=8080)