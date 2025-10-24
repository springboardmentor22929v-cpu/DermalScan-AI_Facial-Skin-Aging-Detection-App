import os, argparse, matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

def main(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    counter = Counter()

    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        n_imgs = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        counter[cls] = n_imgs

    if not counter:
        print(f"No images found under {root}. Please check the path.")
        return

    print("Found classes and counts:")
    for k, v in counter.items():
        print(f"{k}: {v}")

    # Save report CSV
    df = pd.DataFrame(list(counter.items()), columns=["class", "count"])
    csv_path = os.path.join(out_dir, "class_distribution.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(df["class"], df["count"], color="skyblue")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"))
    plt.close()

    print(f"✅ Report saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.root, args.out)
