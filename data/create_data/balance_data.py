import os
import glob
import random

def balance_probing_data(input_dir, output_dir, min_sentences=300):
    """
    Balances SentEval tasks by ensuring VA and TE sets have a minimum number of samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))

    for file_path in txt_files:
        base_name = os.path.basename(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        groups = {"tr": [], "va": [], "te": [], "other": []}
        for line in lines:
            prefix = line[:2]
            if prefix in groups:
                groups[prefix].append(line)
            else:
                groups["other"].append(line)

        random.shuffle(groups["tr"])

        # Reallocate from TR to VA and TE if needed
        for target in ["va", "te"]:
            while len(groups[target]) < min_sentences and groups["tr"]:
                line = groups["tr"].pop()
                groups[target].append(target + line[2:])

        # Final write
        final_lines = groups["tr"] + groups["va"] + groups["te"] + groups["other"]
        with open(os.path.join(output_dir, base_name), "w", encoding="utf-8") as f:
            f.writelines(final_lines)
            
        print(f"Balanced {base_name}: TR={len(groups['tr'])}, VA={len(groups['va'])}, TE={len(groups['te'])}")

if __name__ == "__main__":
    balance_probing_data(
        input_dir="Romanian/merged",
        output_dir="balanced/probing"
    )