import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print("Loading dataset from Hugging Face...")
dataset = load_dataset('microsoft/cats_vs_dogs', split='train')

cats_x, dogs_x, cats_rgb, dogs_rgb = [], [], [], []

print("Processing images into matrices...")
for item in dataset:
    if len(cats_x) >= 1000 and len(dogs_x) >= 1000:
        break
        
    lbl = item['labels']
    x = np.array(item['image'].convert('L').resize((50, 50))).flatten()
    rgb = np.array(item['image'].convert('RGB').resize((200, 200)))
    
    if lbl == 0 and len(cats_x) < 1000: 
        cats_x.append(x)
        cats_rgb.append(rgb)
    elif lbl == 1 and len(dogs_x) < 1000:
        dogs_x.append(x)
        dogs_rgb.append(rgb)

# Standardize and build datasets
X = np.array(cats_x + dogs_x) / 255.0
X_rgb = np.array(cats_rgb + dogs_rgb)
y = np.array([0]*1000 + [1]*1000)

# Split dataset
X_train, X_test, rgb_train, rgb_test, y_train, y_test = train_test_split(
    X, X_rgb, y, test_size=0.2, random_state=42
)

# Apply PCA feature extraction to vastly improve SVM accuracy on raw pixels
print("Applying Principal Component Analysis to improve accuracy...")
pca = PCA(n_components=150, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Train the SVM with tuned Hyperparameters (C=5.0)
print("Training the SVM Model...")
model = SVC(kernel='rbf', C=5.0, gamma='scale')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# Extract sample indices where the model successfully predicted the correct value
correct_idx = [i for i in range(len(y_test)) if y_pred[i] == y_test[i]]

# Paint the graphical window
print("Opening the display window...")
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    idx = correct_idx[i]
    ax.imshow(rgb_test[idx])
    lbl_pred = 'Cat' if y_pred[idx] == 0 else 'Dog'
    lbl_true = 'Cat' if y_test[idx] == 0 else 'Dog'
    ax.set_title(f"Pred: {lbl_pred} | True: {lbl_true}\n[CORRECT]", color="green", fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('svm_predictions.png', bbox_inches='tight')
