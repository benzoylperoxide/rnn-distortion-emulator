import argparse
from scipy.io import wavfile
import numpy as np
import os

def save_wav(filepath, data):
    """Save the WAV file to the specified filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    wavfile.write(filepath, 48000, data.flatten().astype(np.float32))


def normalize(data):
    """Normalize the audio data."""
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max, abs(data_min))
    if data_norm == 0:
        print("[WARNING]: Audio file contains only 0's. Check your input files.")
    return data / data_norm


def contiguous_split(input_data, target_data, train_ratio=0.8, val_ratio=0.1):
    """Split data into contiguous train, val, and test sets."""
    total_samples = len(input_data)

    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    train_input = input_data[:train_end]
    train_target = target_data[:train_end]

    val_input = input_data[train_end:val_end]
    val_target = target_data[train_end:val_end]

    test_input = input_data[val_end:]
    test_target = target_data[val_end:]

    return train_input, train_target, val_input, val_target, test_input, test_target


def process_wav(name, in_file, out_file, normalize_flag):
    """Process the input and target WAV files and save them into train/val/test folders."""
    in_rate, in_data = wavfile.read(in_file)
    out_rate, out_data = wavfile.read(out_file)

    if in_rate != out_rate:
        raise ValueError("Input and output files must have the same sampling rate.")

    # Trim to match lengths
    min_length = min(len(in_data), len(out_data))
    in_data, out_data = in_data[:min_length], out_data[:min_length]

    # Convert to mono if stereo
    if len(in_data.shape) > 1:
        in_data = in_data[:, 0]
    if len(out_data.shape) > 1:
        out_data = out_data[:, 0]

    # Normalize
    if normalize_flag:
        in_data = normalize(in_data)
        out_data = normalize(out_data)

    # Perform contiguous split
    train_input, train_target, val_input, val_target, test_input, test_target = contiguous_split(
        in_data, out_data
    )

    # Save data into respective folders
    base_dir = "data"
    splits = {
        "train": (train_input, train_target),
        "val": (val_input, val_target),
        "test": (test_input, test_target),
    }

    for split_name, (split_input, split_target) in splits.items():
        save_wav(os.path.join(base_dir, split_name, f"{name}_input.wav"), split_input)
        save_wav(os.path.join(base_dir, split_name, f"{name}_target.wav"), split_target)

    print(f"Data split and saved in {base_dir}/train, {base_dir}/val, {base_dir}/test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WAV files for training")
    parser.add_argument("name", help="Output file name prefix")
    parser.add_argument("in_file", help="Input clean WAV file path")
    parser.add_argument("out_file", help="Input target WAV file path")
    parser.add_argument("--normalize", "-n", action="store_true", default=True, help="Normalize WAV files (default: true)")

    args = parser.parse_args()

    process_wav(args.name, args.in_file, args.out_file, args.normalize)
