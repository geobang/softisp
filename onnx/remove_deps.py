import os
import re

def strip_fields_in_file(filepath):
    """Remove deps, needs, provides assignments from a microblock class file."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Match lines like: deps = [...], needs = [...], provides = [...]
        if re.match(r'^\s*(deps|needs|provides)\s*=', line):
            continue  # skip these lines
        new_lines.append(line)

    with open(filepath, "w") as f:
        f.writelines(new_lines)
    print(f"Processed {filepath}")

def strip_fields_in_dir(directory):
    """Process all .py files in a directory recursively."""
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".py"):
                strip_fields_in_file(os.path.join(root, fname))

if __name__ == "__main__":
    # Adjust this path to point to your microblocks source directory
    MICROBLOCKS_DIR = "./microblocks"
    strip_fields_in_dir(MICROBLOCKS_DIR)
