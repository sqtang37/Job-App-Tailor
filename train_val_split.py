import os
import json
import random

DATA_DIR = "data"
TRAIN_OUT = "data/train.jsonl"
VAL_OUT = "data/val.jsonl"

INSTRUCTION_TEXT = (
    "Generate a personalized cover letter tailored to the job posting "
    "using the candidate's resume."
)

TRAIN_RATIO = 0.9
RANDOM_SEED = 42


def is_valid_sample(obj):
    """只保留 output 是 string 的样本"""
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("output"), str)
        and "resume" in obj
        and "job_description" in obj
    )


def load_and_filter_jsonl(data_dir):
    samples = []

    for fname in os.listdir(data_dir):
        if not fname.endswith(".jsonl"):
            continue

        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if is_valid_sample(obj):
                    obj["instruction"] = INSTRUCTION_TEXT
                    samples.append(obj)
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    return samples


def train_val_split(samples, train_ratio=0.9, seed=42):
    random.seed(seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    return samples[:split_idx], samples[split_idx:]


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for obj in samples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


samples = load_and_filter_jsonl(DATA_DIR)
print(f"Total valid samples (output is string): {len(samples)}")

train_samples, val_samples = train_val_split(samples, TRAIN_RATIO, RANDOM_SEED)

write_jsonl(TRAIN_OUT, train_samples)
write_jsonl(VAL_OUT, val_samples)

print(f"Train samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")
print("Done!")
