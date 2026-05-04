import io
from collections import defaultdict

def check_split_coverage(fpath):
    """
    Sometimes erros can happen if in the test or dev data their are unseen words. This gets fixed by putting these sentences in the training set.
    """
    tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}

    # Count labels per split
    label_counts = defaultdict(lambda: {'train': 0, 'dev': 0, 'test': 0})

    with io.open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split('\t')
            split = tok2split[parts[0]]
            label = parts[1]
            label_counts[label][split] += 1

    # Find problematic labels
    problematic = {}
    total_lines_to_change = 0

    for label, counts in label_counts.items():
        if counts['train'] == 0 and (counts['dev'] > 0 or counts['test'] > 0):
            lines_needed = counts['dev'] + counts['test']
            problematic[label] = lines_needed
            total_lines_to_change += 1  # Only need to move ONE per label

    print("Labels missing in training set:")
    for label, lines in problematic.items():
        print(f"  Label '{label}' → {lines} lines in dev/test")

    print("\nNumber of labels that require fixing:", len(problematic))
    print("Minimum number of lines to change (move to train):", total_lines_to_change)

    return problematic

def fix_split_coverage(input_path, output_path):
    tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
    split2tok = {'train': 'tr', 'dev': 'va', 'test': 'te'}

    # First pass: count labels per split
    label_counts = defaultdict(lambda: {'train': 0, 'dev': 0, 'test': 0})
    lines = []

    with io.open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n')
            fields = parts.split('\t')
            split = tok2split[fields[0]]
            label = fields[1]

            label_counts[label][split] += 1
            lines.append(fields)

    # Identify labels missing in train
    labels_to_fix = {
        label for label, counts in label_counts.items()
        if counts['train'] == 0 and (counts['dev'] > 0 or counts['test'] > 0)
    }

    print("Labels needing fix:", len(labels_to_fix))

    moved = set()

    # Second pass: move one example per problematic label
    for fields in lines:
        split = tok2split[fields[0]]
        label = fields[1]

        if label in labels_to_fix and label not in moved:
            # Change split to train
            fields[0] = split2tok['train']
            moved.add(label)

        if len(moved) == len(labels_to_fix):
            break

    # Write fixed file
    with io.open(output_path, 'w', encoding='utf-8') as f:
        for fields in lines:
            f.write('\t'.join(fields) + '\n')

    print("Lines moved to train:", len(moved))
    print("Fixed file written to:", output_path)


check_split_coverage("Romanian/merged/word_content.txt")
fix_split_coverage("merged/word_content.txt", "merged/word_content_fixed.txt")
check_split_coverage("merged/word_content_fixed.txt")