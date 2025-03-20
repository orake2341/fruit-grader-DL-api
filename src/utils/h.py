import os


def rename_files(directory):
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)

        # Skip if the filename is too short or already formatted
        if len(name) < 2 or name[-2] == "(" and name[-1] == ")":
            continue

        new_name = f"{name[:-1]}({name[-1]}){ext}"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")


# Change 'your_directory_path' to the actual folder path
directory_path = "../../data/Dataset/labels"
rename_files(directory_path)
