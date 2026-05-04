import os
import glob
import random
"""
This code is used to combine two SentEval datasets into one.
"""

#example for the romanian datasets
rrt_folder = "Romanian_RRT/"
nonstandard_folder = "Romanian_Nonstandard/"
merged_folder = "SentEval_Data/Romanian/merged/"
os.makedirs(merged_folder, exist_ok=True)

# Get sorted file lists (must align corresponding files)
rrt_files = sorted(glob.glob(rrt_folder + "*.txt"))
nonstandard_files = sorted(glob.glob(nonstandard_folder + "*.txt"))

# Ensure same number of files
assert len(rrt_files) == len(nonstandard_files), "Mismatch in number of files!"

for r_file, n_file in zip(rrt_files, nonstandard_files):
    # Output merged filename
    base_name = os.path.basename(r_file)
    merged_file = os.path.join(merged_folder, base_name)

    # Read both files
    merged_lines = []
    for f in [r_file, n_file]:
        with open(f, "r", encoding="utf-8") as in_f:
            merged_lines.extend(in_f.readlines())

    # Group by prefix: tr → va → te → others
    groups = {"tr": [], "va": [], "te": [], "other": []}
    for line in merged_lines:
        stripped = line.strip()
        if stripped.startswith("tr"):
            groups["tr"].append(line)
        elif stripped.startswith("va"):
            groups["va"].append(line)
        elif stripped.startswith("te"):
            groups["te"].append(line)
        else:
            groups["other"].append(line)

    # Shuffle within each group
    for g in groups:
        random.shuffle(groups[g])

    # Write groups in order: tr → va → te → other
    final_lines = groups["tr"] + groups["va"] + groups["te"] + groups["other"]
    with open(merged_file, "w", encoding="utf-8") as out_f:
        out_f.writelines(final_lines)

    print(f"Merged: {base_name} → TR={len(groups['tr'])}, VA={len(groups['va'])}, TE={len(groups['te'])}")

print("Merged and shuffled all files into:", merged_folder)