import pandas as pd
import re

# Load your CSV file
csv_file = "../../data/freshness_grading.csv"  # Replace with your actual file name
df = pd.read_csv(csv_file)


# Function to extract fruit type from the filename
def get_fruit_type(filename):
    match = re.match(
        r"([a-zA-Z]+)\d+", filename
    )  # Extracts letters before the first digit
    if match:
        return match.group(
            1
        ).capitalize()  # Return the fruit type with the first letter capitalized
    return "Unknown"


# Apply the function to create a new column "fruit type"
df["fruit_type"] = df["image_name"].apply(get_fruit_type)

# Save the updated CSV file
updated_csv_file = "fresh.csv"  # Replace with your desired output file name
df.to_csv(updated_csv_file, index=False)

print(f"New column 'fruit type' added. Updated file saved as {updated_csv_file}.")
