import os


def update_class_label_in_folder(folder_path, old_class=16, new_class=15):
    """Update class label in all YOLO annotation files inside a folder."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Process only .txt files
            txt_path = os.path.join(folder_path, filename)

            with open(txt_path, "r") as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                values = line.strip().split()
                if (
                    int(values[0]) == old_class
                ):  # Change only if class matches old_class
                    values[0] = str(new_class)
                updated_lines.append(" ".join(values) + "\n")

            # Overwrite the file with updated content
            with open(txt_path, "w") as f:
                f.writelines(updated_lines)
            print(f"Updated: {filename}")


# Example usage
folder_path = (
    "../../data/Dataset/valid/labels"  # Change this to your actual folder path
)
update_class_label_in_folder(folder_path)
