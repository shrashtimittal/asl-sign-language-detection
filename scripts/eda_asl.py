import os, random
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# --------- change this path if your folder name is different -----------
DATA_DIR = r"E:/ASL_Project/raw/archive/asl_alphabet_train/asl_alphabet_train"
# -----------------------------------------------------------------------

SAMPLES_PER_CLASS = 6
OUT_IMG = r"E:/ASL_Project/eda_samples.png"

def find_classes_and_files(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    files = {c: glob(os.path.join(root, c, "*")) for c in classes}
    return classes, files

def print_counts(files):
    counts = {c: len(files[c]) for c in files}
    print(f"Total classes: {len(counts)}")
    print("Top 10 classes by count:")
    for c,n in sorted(counts.items(), key=lambda x:-x[1])[:10]:
        print(f"{c:12s} : {n}")
    return counts

def save_sample_grid(files, classes, per_class=6, out=OUT_IMG):
    cols = per_class
    rows = min(len(classes), 10)   # show up to 10 classes
    fig = plt.figure(figsize=(cols*1.8, rows*1.8))
    for i,c in enumerate(classes[:rows]):
        imgs = files[c]
        sample = random.sample(imgs, min(per_class, len(imgs)))
        for j,img_path in enumerate(sample):
            ax = fig.add_subplot(rows, cols, i*cols + j + 1)
            ax.axis('off')
            ax.imshow(Image.open(img_path).convert('RGB'))
            if j == 0: ax.set_title(c, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved grid image to {out}")

if __name__ == "__main__":
    classes, files = find_classes_and_files(DATA_DIR)
    counts = print_counts(files)
    save_sample_grid(files, classes, per_class=SAMPLES_PER_CLASS)
