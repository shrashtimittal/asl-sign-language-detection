import os, shutil
from glob import glob
from sklearn.model_selection import train_test_split

# ---- adjust if your class folders are elsewhere ----
SRC_DIR = r"E:/ASL_Project/raw/archive/asl_alphabet_train/asl_alphabet_train"
DST_DIR = r"E:/ASL_Project/dataset_split"
VAL_RATIO = 0.1   # 10% for validation
TEST_RATIO = 0.1  # 10% for test
SEED = 42
# ----------------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_and_copy():
    classes = sorted([d for d in os.listdir(SRC_DIR)
                      if os.path.isdir(os.path.join(SRC_DIR, d))])
    for c in classes:
        files = glob(os.path.join(SRC_DIR, c, "*"))
        trainval, test = train_test_split(files,
                                          test_size=TEST_RATIO,
                                          random_state=SEED)
        train, val = train_test_split(trainval,
                                      test_size=VAL_RATIO/(1-TEST_RATIO),
                                      random_state=SEED)

        for split_name, split_files in [("train", train),
                                        ("val",   val),
                                        ("test",  test)]:
            out_dir = os.path.join(DST_DIR, split_name, c)
            ensure_dir(out_dir)
            for f in split_files:
                shutil.copy2(f, os.path.join(out_dir, os.path.basename(f)))
        print(f"{c:3s} -> train:{len(train)} val:{len(val)} test:{len(test)}")
    print("✅ Split completed at:", DST_DIR)

if __name__ == "__main__":
    split_and_copy()
