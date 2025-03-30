import hashlib
import os


def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def find_duplicate_images(train_dir, valid_dir):
    train_hashes = {
        get_file_hash(os.path.join(train_dir, f)) for f in os.listdir(train_dir)
    }
    valid_hashes = {
        get_file_hash(os.path.join(valid_dir, f)) for f in os.listdir(valid_dir)
    }

    duplicates = train_hashes.intersection(valid_hashes)
    return duplicates


train_dir = "../../data/Dataset/train/images"
valid_dir = "../../data/Dataset/valid/images"

duplicates = find_duplicate_images(train_dir, valid_dir)

if duplicates:
    print(f"Found {len(duplicates)} duplicate images.")
else:
    print("No duplicate images found.")
