import argparse
import numpy as np
import torch
from CoreAudioML import miscfuncs, training, dataset as CAMLdataset, networks
from torch.utils.tensorboard import SummaryWriter
import os
import time
from scipy.io.wavfile import write
import csv

def init_model(save_path, args):
    """Initialize the model, either loading from saved state or creating a new one."""
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    if miscfuncs.file_check('model.json', save_path) and args.load_model:
        print('Loading existing model.')
        model_data = miscfuncs.json_load('model', save_path)
        network = networks.load_model(model_data)
    else:
        print('Creating new network.')
        network = networks.SimpleRNN(
            input_size=args.input_size,
            unit_type=args.unit_type,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            skip=args.skip
        )
        network.save_state = False
        network.save_model('model', save_path)
    return network

def main(args):
    """Main function to handle training and testing."""
    save_path = os.path.join(args.save_location, args.name)
    network = init_model(save_path, args)

    if not torch.cuda.is_available() or not args.cuda:
        print('CUDA not available or disabled.')
        cuda = False
    else:
        print('CUDA enabled.')
        network = network.cuda()
        cuda = True

    optimiser = torch.optim.Adam(network.parameters(), lr=args.learn_rate, weight_decay=1e-4)
    loss_functions = training.LossWrapper({'ESR': 0.75, 'DC': 0.25})
    train_track = training.TrainTrack()
    writer = SummaryWriter(os.path.join('TensorboardData', args.name))

    dataset = CAMLdataset.DataSet(data_dir='data')
    dataset.create_subset('train', frame_len=22050)
    dataset.load_file(f"train/{args.name}", 'train')  # Training data path
    dataset.create_subset('val')
    dataset.load_file(f"val/{args.name}", 'val')  # Validation data path

    print(f"Training {args.model} model on {args.name}")
    start_time = time.time()

    # Initialize CSV for logging epoch-loss data
    csv_path = os.path.join(save_path, "epoch_loss.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(["Epoch", "Loss"])  # Write CSV header

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            epoch_loss = network.train_epoch(
                dataset.subsets['train'].data['input'][0],
                dataset.subsets['train'].data['target'][0],
                loss_functions,
                optimiser,
                args.batch_size,
                200,
                1000
            )
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time

            writer.add_scalar('Training/TrainingLoss', epoch_loss, epoch)
            writer_csv.writerow([epoch, epoch_loss])  # Log epoch and loss to CSV
            file.flush()  # Ensure data is written immediately

            print(f"Epoch {epoch}/{args.epochs} | Loss: {epoch_loss:.6f} | "
                  f"Epoch Time: {epoch_time:.2f}s | Elapsed Time: {total_time:.2f}s")

    print("Training complete. Total time: {:.2f}s".format(total_time))

    # Save the best model as JSON
    print("Saving the best model...")
    model_data = {
        "model_data": {
            "model": args.model,
            "input_size": args.input_size,
            "skip": args.skip,
            "output_size": args.output_size,
            "unit_type": args.unit_type,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "bias_fl": True,
        },
        "state_dict": {k: v.cpu().tolist() for k, v in network.state_dict().items()},
    }
    json_path = os.path.join(save_path, "not_model_best.json")
    with open(json_path, "w") as f:
        import json
        json.dump(model_data, f, indent=4)
    print(f"Model saved to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN models for audio processing.")
    parser.add_argument("name", help="Model name")
    parser.add_argument("--model", default="SimpleRNN", help="Model architecture")
    parser.add_argument("--input_size", type=int, default=1, help="Input size")
    parser.add_argument("--output_size", type=int, default=1, help="Output size")
    parser.add_argument("--unit_type", default="GRU", help="RNN unit type (LSTM, GRU)")
    parser.add_argument("--hidden_size", type=int, default=10, help="Hidden size of RNN unit")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--skip", type=int, default=1, help="Skip connections")
    parser.add_argument("--cuda", type=bool, default=True, help="Enable CUDA")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--learn_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--save_location", default="results", help="Directory to save models")
    parser.add_argument("--load_model", type=bool, default=True, help="Load existing model if available")

    args = parser.parse_args()
    main(args)
