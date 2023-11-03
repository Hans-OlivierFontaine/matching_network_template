from datasets import MatchingDataset
from experiments.NShotDatasetBuilder import NShotDatasetBuilder
import tqdm
import argparse
from pathlib import Path
from datetime import datetime
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, default="/home/fontaine_ha/shares/data/PlateformeHP/LORD/PROJETS_EN_COURS/Projet_Weed_Sentinel/data/DeepWeeds/", help="Root directory of dataset")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--fce", type=bool, default=True, help="Use a fully connected network")
    parser.add_argument("--n_ways", type=int, default=5, help="Classes per set")
    parser.add_argument("--k_shots", type=int, default=5, help="Samples per class")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--img_size", type=int, default=256, help="Size of images")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--total_train_batches", type=int, default=100, help="Number of train batches")
    parser.add_argument("--total_val_batches", type=int, default=100, help="Number of validation batches")
    parser.add_argument("--total_test_batches", type=int, default=250, help="Number of epochs")
    parser.add_argument("--outputs_dir", type=Path,
                        default="/home/fontaine_ha/shares/data/PlateformeHP/LORD/PROJETS_EN_COURS/Projet_Weed_Sentinel/data/static_output_matching/",
                        help="Output directory")
    return parser.parse_args()


def define_output(outputs_dir: Path) -> Path:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder_path = outputs_dir / current_datetime
    output_folder_path.mkdir(exist_ok=True, parents=True)
    return output_folder_path


def main(args):
    log_dir = define_output(args.outputs_dir)
    log_dir = log_dir / f'Matching-BS{args.batch_size}-FCE{args.fce}-WAYS{args.n_ways}-SHOTS{args.k_shots}-channels{args.channels}'
    logging.basicConfig(filename=log_dir.absolute().__str__(), level=logging.INFO)
    logging.info(f"setup:{args.__str__()}")
    data_train = MatchingDataset.MatchingDataset(dataroot=args.dataroot, version='train',
                                                 n_episodes=args.total_train_batches * args.batch_size,
                                                 n_ways=args.n_ways, k_shots=args.k_shots, img_size=args.img_size,
                                                 channels=args.channels)
    data_val = MatchingDataset.MatchingDataset(dataroot=args.dataroot, version='val',
                                               n_episodes=args.total_train_batches * args.batch_size,
                                               n_ways=args.n_ways, k_shots=args.k_shots, img_size=args.img_size,
                                               channels=args.channels)
    data_test = MatchingDataset.MatchingDataset(dataroot=args.dataroot, version='test',
                                                n_episodes=args.total_train_batches * args.batch_size,
                                                n_ways=args.n_ways, k_shots=args.k_shots, img_size=args.img_size,
                                                channels=args.channels)
    dataset_build = NShotDatasetBuilder(data_train=data_train, data_val=data_val, data_test=data_test,
                                        batch_size=args.batch_size, n_ways=args.n_ways, k_shots=args.k_shots,
                                        channels=args.channels, fce=args.fce, img_size=args.img_size)
    best_val = 0.
    with tqdm.tqdm(total=args.epochs) as pbar_e:
        for e in range(0, args.epochs):
            total_c_loss, total_accuracy = dataset_build.run_training_epoch()
            # print("Epoch {}: train_loss: {}, train_accuracy: {}%".format(e, round(total_c_loss, 4), round(total_accuracy * 100, 4)))
            total_val_c_loss, total_val_accuracy = dataset_build.run_validation_epoch()
            # print("Epoch {}: val_loss: {}, val_accuracy: {}%".format(e, round(total_val_c_loss, 4), round(total_val_accuracy * 100, 4)))
            logging.info(f'train_loss: {round(total_c_loss, 4)}')
            logging.info(f'train_acc: {round(total_accuracy * 100, 4)}%')
            logging.info(f'val_loss: {round(total_val_c_loss, 4)}')
            logging.info(f'val_acc: {round(total_val_accuracy * 100, 4)}%')
            if total_val_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
                best_val = total_val_accuracy
                total_test_c_loss, total_test_accuracy = dataset_build.run_testing_epoch()
                # print("Epoch {}: test_loss: {}, test_accuracy: {}%".format(e, round(total_test_c_loss, 4), round(total_test_accuracy * 100, 4)))
                logging.info(f'test_loss: {round(total_test_c_loss, 4)}')
                logging.info(f'test_acc: {round(total_test_accuracy, 4)}%')
            else:
                total_test_c_loss = -1
                total_test_accuracy = -1
            pbar_e.update(1)


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
