import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV containing image names & labels
df = pd.read_csv("../../data/Dataset/Vgg11/banana.csv")

# Separate images by video
video1_data = df[df["image_name"].str.contains(r"\(a\)", regex=True)]
video2_data = df[df["image_name"].str.contains(r"\(b\)", regex=True)]
video3_data = df[df["image_name"].str.contains(r"\(c\)", regex=True)]

# Combine Video 1 & 2 for training (840 images)
train_data = pd.concat([video1_data, video2_data])

# Reduce Video 3 to 210 images
video3_selected = video3_data.sample(n=175, random_state=42)

# Split Video 3 into validation (105) and test (105)
val_data, test_data = train_test_split(video3_selected, test_size=0.5, random_state=42)

# Save new CSV with updated dataset
updated_df = pd.concat([train_data, val_data, test_data])
updated_df.to_csv("../../data/Dataset/Vgg11/banana_updated.csv", index=False)

print(
    f"Final dataset sizes -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
)
