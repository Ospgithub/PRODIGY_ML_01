"""
Food Recognition & Calorie Estimation  |  Task 5 – Prodigy ML Internship
--------------------------------------------------------------------------
Features:
  - Uses pre-trained EfficientNetB0 (no training/downloading dataset required)
  - Opens a file dialog to select an image or accepts via CLI
  - Recognizes 20+ common foods and maps to calorie/macronutrient data
  - Displays a rich 3-panel visualization: Image | Confidence | Nutrition

Install : pip install torch torchvision matplotlib Pillow
Run     : python Model.py
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
#  Nutrition Database (mapping ImageNet food classes to macros)
# ─────────────────────────────────────────────────────────────────────────────
NUTRITION = {
    "pizza":           {"label": "Pizza (1 slice)",          "kcal": 285, "protein": 12.6, "carbs": 35.7, "fat": 9.8, "emoji": "🍕", "color": "#e74c3c"},
    "cheeseburger":    {"label": "Cheeseburger (1 burger)",  "kcal": 540, "protein": 25.8, "carbs": 40.0, "fat": 28.5, "emoji": "🍔", "color": "#e67e22"},
    "hot dog":         {"label": "Hot Dog (1 piece)",        "kcal": 290, "protein": 10.3, "carbs": 23.8, "fat": 17.2, "emoji": "🌭", "color": "#f39c12"},
    "ice cream":       {"label": "Ice Cream (1 scoop)",      "kcal": 207, "protein": 3.5,  "carbs": 23.6, "fat": 11.0, "emoji": "🍦", "color": "#9b59b6"},
    "guacamole":       {"label": "Guacamole (2 tbsp)",       "kcal": 45,  "protein": 0.6,  "carbs": 2.4,  "fat": 4.1,  "emoji": "🥑", "color": "#2ecc71"},
    "bagel":           {"label": "Bagel (1 medium)",         "kcal": 245, "protein": 10.0, "carbs": 48.0, "fat": 1.5,  "emoji": "🥯", "color": "#f1c40f"},
    "pretzel":         {"label": "Pretzel (1 medium)",       "kcal": 389, "protein": 10.9, "carbs": 80.0, "fat": 2.6,  "emoji": "🥨", "color": "#e67e22"},
    "mashed potato":   {"label": "Mashed Potato (1 cup)",    "kcal": 214, "protein": 4.0,  "carbs": 35.0, "fat": 7.0,  "emoji": "🥔", "color": "#f1c40f"},
    "broccoli":        {"label": "Broccoli (1 cup chopped)", "kcal": 31,  "protein": 2.6,  "carbs": 6.0,  "fat": 0.3,  "emoji": "🥦", "color": "#27ae60"},
    "cauliflower":     {"label": "Cauliflower (1 cup)",      "kcal": 27,  "protein": 2.1,  "carbs": 5.3,  "fat": 0.3,  "emoji": "🥦", "color": "#ecf0f1"},
    "strawberry":      {"label": "Strawberry (1 cup)",       "kcal": 49,  "protein": 1.0,  "carbs": 11.7, "fat": 0.5,  "emoji": "🍓", "color": "#e74c3c"},
    "orange":          {"label": "Orange (1 medium)",        "kcal": 62,  "protein": 1.2,  "carbs": 15.4, "fat": 0.2,  "emoji": "🍊", "color": "#e67e22"},
    "lemon":           {"label": "Lemon (1 medium)",         "kcal": 17,  "protein": 0.6,  "carbs": 5.4,  "fat": 0.2,  "emoji": "🍋", "color": "#f1c40f"},
    "banana":          {"label": "Banana (1 medium)",        "kcal": 105, "protein": 1.3,  "carbs": 27.0, "fat": 0.4,  "emoji": "🍌", "color": "#f1c40f"},
    "pineapple":       {"label": "Pineapple (1 cup chunks)", "kcal": 82,  "protein": 0.9,  "carbs": 21.6, "fat": 0.2,  "emoji": "🍍", "color": "#f1c40f"},
    "pomegranate":     {"label": "Pomegranate (1/2 cup)",    "kcal": 72,  "protein": 1.5,  "carbs": 16.3, "fat": 1.0,  "emoji": "🍎", "color": "#c0392b"},
    "meatloaf":        {"label": "Meatloaf (1 slice)",       "kcal": 149, "protein": 12.0, "carbs": 7.0,  "fat": 8.0,  "emoji": "🥩", "color": "#c0392b"},
    "burrito":         {"label": "Burrito (1 medium)",       "kcal": 430, "protein": 16.0, "carbs": 50.0, "fat": 18.0, "emoji": "🌯", "color": "#e67e22"},
    "espresso":        {"label": "Espresso (1 shot)",        "kcal": 3,   "protein": 0.1,  "carbs": 0.5,  "fat": 0.0,  "emoji": "☕", "color": "#7f8c8d"},
    "cup":             {"label": "Cup of Coffee/Tea",        "kcal": 2,   "protein": 0.0,  "carbs": 0.0,  "fat": 0.0,  "emoji": "🍵", "color": "#95a5a6"},
    "carbonara":       {"label": "Carbonara (1 serving)",    "kcal": 595, "protein": 22.0, "carbs": 60.0, "fat": 28.0, "emoji": "🍝", "color": "#f39c12"},
    "french loaf":     {"label": "Baguette (1 slice)",       "kcal": 150, "protein": 5.0,  "carbs": 30.0, "fat": 1.0,  "emoji": "🥖", "color": "#d35400"},
    "apple":           {"label": "Apple (1 medium)",         "kcal": 95,  "protein": 0.5,  "carbs": 25.0, "fat": 0.3,  "emoji": "🍏", "color": "#2ecc71"},
    "bell pepper":     {"label": "Bell Pepper (1 medium)",   "kcal": 24,  "protein": 1.0,  "carbs": 6.0,  "fat": 0.2,  "emoji": "🫑", "color": "#e74c3c"},
    "mushroom":        {"label": "Mushroom (1 cup)",         "kcal": 15,  "protein": 2.2,  "carbs": 2.3,  "fat": 0.2,  "emoji": "🍄", "color": "#bdc3c7"},
    "pot pie":         {"label": "Pot Pie (1 piece)",        "kcal": 400, "protein": 14.0, "carbs": 38.0, "fat": 22.0, "emoji": "🥧", "color": "#f39c12"},
    "red wine":        {"label": "Red Wine (1 glass)",       "kcal": 125, "protein": 0.1,  "carbs": 3.8,  "fat": 0.0,  "emoji": "🍷", "color": "#8e44ad"},
    "soup":            {"label": "Soup / Broth (1 bowl)",    "kcal": 150, "protein": 8.0,  "carbs": 15.0, "fat": 5.0,  "emoji": "🥣", "color": "#e67e22"},
}

# ImageNet class indices mapped to our named keys
IMAGENET_CLASSES = {
    963: "pizza", 933: "cheeseburger", 934: "hot dog", 928: "ice cream",
    924: "guacamole", 931: "bagel", 932: "pretzel", 935: "mashed potato",
    937: "broccoli", 938: "cauliflower", 949: "strawberry", 950: "orange",
    951: "lemon", 954: "banana", 953: "pineapple", 957: "pomegranate",
    962: "meatloaf", 965: "burrito", 967: "espresso", 968: "cup",
    959: "carbonara", 930: "french loaf", 948: "apple", 945: "bell pepper",
    947: "mushroom", 964: "pot pie", 966: "red wine", 925: "soup", 926: "soup",
    929: "ice cream"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Pre-trained Model Setup
# ─────────────────────────────────────────────────────────────────────────────
print("  ⬇  Loading EfficientNetB0 pre-trained model...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.eval()

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

def predict_image(img_path: str):
    img = Image.open(img_path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
    
    probs = torch.nn.functional.softmax(out[0], dim=0)
    
    # Get top 3 predictions
    top_prob, top_catid = torch.topk(probs, 3)
    top_prob = top_prob.numpy()
    top_catid = top_catid.numpy()

    # Find the top predicted *food* class from our database
    food_probs = {}
    best_food_id = None
    best_food_prob = -1

    for i in range(len(probs)):
        i_prob = probs[i].item()
        if i in IMAGENET_CLASSES:
            food_name = IMAGENET_CLASSES[i]
            food_probs[food_name] = i_prob
            if i_prob > best_food_prob:
                best_food_prob = i_prob
                best_food_id = i

    if best_food_id is None:
        best_food_name = "Unknown Food"
        best_food_prob = 0.0
    else:
        best_food_name = IMAGENET_CLASSES[best_food_id]

    # Instead of discarding food if a plate/table is predicted #1,
    # we consider it a confident food prediction if the top food probability is > 1.5%.
    is_food = best_food_prob > 0.015

    return {
        "image": img,
        "class": best_food_name,
        "is_food_confident": is_food,
        "confidence": best_food_prob,
        "probs": food_probs,
        "nutrition": NUTRITION.get(best_food_name, None)
    }

# ─────────────────────────────────────────────────────────────────────────────
#  Visualization  (3-panel matplotlib figure)
# ─────────────────────────────────────────────────────────────────────────────
_BG    = "#0d1117"
_PANEL = "#161b22"
_TEXT  = "#e6edf3"
_SUB   = "#8b949e"
_DIM   = "#21262d"

def show_results(result: dict) -> None:
    cls = result["class"]
    nut = result["nutrition"]
    conf = result["confidence"]
    probs = result["probs"]

    fig = plt.figure(figsize=(16, 7.4), facecolor=_BG)
    
    if not nut or not result["is_food_confident"]:
        print("\n  [WARN] Image might not be a supported food item or confidence is too low.")
        plt.imshow(result["image"])
        plt.title("Not recognized as a supported food in our database.", color='white')
        fig.patch.set_facecolor(_DIM)
        plt.axis("off")
        plt.show()
        return

    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.35, 1.0, 0.88],
        wspace=0.05, left=0.01, right=0.99,
        top=0.90, bottom=0.07,
    )

    accent = nut.get("color", "#3498db")

    # ── Panel 1 : Food image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(result["image"])
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_edgecolor(accent); sp.set_linewidth(3.0)

    name_str = cls.replace("_", " ").title()
    ax1.set_title(f"{nut['emoji']}  {name_str}", color=_TEXT, fontsize=17, fontweight="bold", pad=8)
    ax1.set_xlabel(f"Confidence  {conf * 100:.1f}%", color=accent, fontsize=12, labelpad=7)

    # ── Panel 2 : Confidence bar chart
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(_PANEL)
    for sp in ax2.spines.values():
        sp.set_visible(False)

    ordered = sorted(probs, key=probs.get, reverse=True)[:5] # Show top 5 foods
    labels  = [f"{NUTRITION.get(c, {}).get('emoji', '🍽️')}  {c.replace('_', ' ').title()}" for c in ordered]
    values  = [probs[c] * 100 for c in ordered]
    colors  = [NUTRITION.get(c, {}).get("color", "#3498db") for c in ordered]
    alphas  = [1.0 if c == cls else 0.28 for c in ordered]

    bars = ax2.barh(labels, values, color=colors, height=0.55, edgecolor="none")
    for bar, alp, val in zip(bars, alphas, values):
        bar.set_alpha(alp)
        fw = "bold" if alp == 1.0 else "normal"
        ax2.text(min(val + 1.5, 109), bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                 va="center", ha="left", color=_TEXT, fontsize=10, fontweight=fw)

    ax2.set_xlim(0, max(max(values) * 1.3, 10))
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", colors=_TEXT, labelsize=10.5)
    ax2.tick_params(axis="x", colors=_SUB,  labelsize=8)
    ax2.set_xlabel("Confidence (%)", color=_SUB, fontsize=10)
    ax2.set_title("Prediction Scores", color=_TEXT, fontsize=13, fontweight="bold", pad=10)
    ax2.set_facecolor(_PANEL)
    ax2.xaxis.grid(True, color=_DIM, linewidth=0.5, linestyle="--")
    ax2.set_axisbelow(True)

    # ── Panel 3 : Nutrition card
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(_PANEL)
    for sp in ax3.spines.values():
        sp.set_visible(False)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 11)
    ax3.set_title("Nutrition Card", color=_TEXT, fontsize=13, fontweight="bold", pad=10)

    # Serving description
    ax3.text(5, 10.35, nut["label"], ha="center", color=_SUB, fontsize=8.5)

    # Big calorie number
    ax3.text(5, 8.8, str(nut["kcal"]), ha="center", color=accent, fontsize=54, fontweight="bold")
    ax3.text(5, 7.35, "kcal  ·  per serving", ha="center", color=_SUB, fontsize=10)
    ax3.axhline(6.85, xmin=0.06, xmax=0.94, color=_SUB, linewidth=0.5, alpha=0.35)

    # Macro bars
    macros = [
        ("Protein",  nut["protein"], "#3498db"),
        ("Carbs",    nut["carbs"],   "#2ecc71"),
        ("Fat",      nut["fat"],     "#e74c3c"),
    ]
    max_val = max(m[1] for m in macros) or 1.0
    MAX_W   = 5.5

    for i, (name, val, col) in enumerate(macros):
        y     = 5.95 - i * 1.55
        bar_w = (val / max_val) * MAX_W
        ax3.barh(y, MAX_W, left=2.3, height=0.5, color=_DIM, edgecolor="none")
        ax3.barh(y, bar_w, left=2.3, height=0.5, color=col, edgecolor="none", alpha=0.90)
        ax3.text(2.15, y, name, ha="right", va="center", color=_SUB, fontsize=9)
        ax3.text(2.3 + bar_w + 0.18, y, f"{val} g", ha="left", va="center", color=_TEXT, fontsize=9, fontweight="bold")

    ax3.text(5, 0.45, "* Estimates for a standard serving size", ha="center", color=_SUB, fontsize=7.2, style="italic")
    fig.suptitle("🍽️  Food Recognition & Calorie Estimation  |  Prodigy ML – Task 5", color=_TEXT, fontsize=13, fontweight="bold", y=0.97)

    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'─' * 56}")
    print("  Please select a food image from the popup dialog...")
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    img_path = filedialog.askopenfilename(
        title="Select Food Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    
    if not img_path:
        print("\n  [INFO] No image selected. Exiting.")
        sys.exit(0)

    if not os.path.isfile(img_path):
        print(f"\n  [ERROR] File not found: {img_path}")
        sys.exit(1)

    print(f"\n  Analysing: {img_path}")
    result = predict_image(img_path)
    n = result["nutrition"]

    if n and result["is_food_confident"]:
        print(f"\n{'═' * 56}")
        print(f"  {n['emoji']}  Prediction  : {result['class'].replace('_', ' ').title()}")
        print(f"      Confidence : {result['confidence'] * 100:.1f}%")
        print(f"      Calories   : {n['kcal']} kcal  ({n['label']})")
        print(f"      Protein    : {n['protein']} g")
        print(f"      Carbs      : {n['carbs']} g")
        print(f"      Fat        : {n['fat']} g")
        print(f"{'═' * 56}\n")
    else:
        print("\n  [INFO] Model could not confidently identify a supported food class in this image.")

    show_results(result)

if __name__ == "__main__":
    main()
