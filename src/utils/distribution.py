import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

# Define paths to label folders
LABELS_PATHS = {
    "Train": "../../data/Dataset/train/labels",
}

# Initialize dictionary to store class counts
total_class_counts = Counter()

# Loop through each dataset folder
for path in LABELS_PATHS.values():
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r") as f:
                for line in f:
                    class_id = int(line.split()[0])  # Extract class ID
                    total_class_counts[class_id] += 1  # Count occurrences

# Optional: Define class names if you have classes.txt
CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

# Create DataFrame for plotting
df_list = [
    {
        "Class": CLASS_NAMES[class_id]
        if class_id < len(CLASS_NAMES)
        else f"Class {class_id}",
        "Count": count,
    }
    for class_id, count in total_class_counts.items()
]

df = pd.DataFrame(df_list)

# Plot combined class distribution
plt.figure(figsize=(12, 6))
sns.barplot(x="Class", y="Count", data=df, palette="coolwarm")

# Add counts on top of each bar
for index, row in enumerate(df.itertuples()):
    plt.text(
        index,
        row.Count + 5,
        str(row.Count),
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

plt.xlabel("Class")
plt.ylabel("Total Number of Instances")
plt.title("Total YOLO Dataset Class Distribution")
plt.xticks(rotation=45)
plt.show()
