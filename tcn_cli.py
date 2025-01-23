from pathlib import Path
import click
import random
import shutil
from fall_detection import extract_and_save_key_frames, extract_keypoints_with_pose_model
from fall_detection import train, evaluate


@click.group()
def tcn():
    """Group for TCN commands."""
    pass


@tcn.command(name='split')
@click.option('--input', type=Path, required=True, help='Path to input video directory.')
@click.option('--output', type=Path, required=True, help='Path to save output directory.')
@click.option('--train_ratio', type=float, required=False, default=0.7, help='Train ratio for splitting the dataset.')
@click.option('--val_ratio', type=float, required=False, default=0.15, help='Validation ratio for splitting the dataset.')
@click.option('--test_ratio', type=float, required=False, default=0.15, help='Test ratio for splitting the dataset.')
def splitDataset(input, output, train_ratio, val_ratio, test_ratio):
    """
    Prepares the dataset by splitting videos into train, val, and test sets.
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    assert sum(ratios) == 1.0, 'Split ratios must sum to 1.0'
    categories = ['ADL', 'FALL']
    random.seed(42)  # Random seed for reproducibility.

    for category in categories:
        category_path = Path(input) / category
        video_files = list(category_path.glob('*.mp4'))
        random.shuffle(video_files)

        train_split = int(train_ratio * len(video_files))
        val_split = int(val_ratio * len(video_files)) + train_split

        splits = {
            'train': video_files[:train_split],
            'val': video_files[train_split:val_split],
            'test': video_files[val_split:]
        }

        for split, videos in splits.items():
            split_dir = Path(output) / split / category
            split_dir.mkdir(parents=True)
            for video in videos:
                # Copy videos
                # destination = split_dir / video.name
                # destination.write_bytes(video.read_bytes())
                shutil.copy(video, split_dir)

    print(f'Dataset prepared at {output}.')


@tcn.command(name='extract')
@click.option('--input', type=Path, required=True, help='Path to the split video directory.')
@click.option('--output', type=Path, required=True, help='Path to save output directory.')
@click.option('--model', type=Path, required=True, help='Path to the Pose Estimation model.')
def extractData(input, output, model):
    """
    Extract key frames from videos in split dataset. Using the provided pose model,the keypoints for each frame will be extracted.
    :param input: Path to the input directory containing `FALL` and `ADL` subdirectories.
    :param output: Path to the output directory to save split datasets
    :param keypoints: If True, a pose estimation model is used to generate keypoints for each frame.
    :param model: Path to the pose estimation model.
    """

    for split in ['train', 'val', 'test']:
        from tqdm import tqdm
        for category in ['ADL', 'FALL']:
            video_dir = Path(input) / split / category
            output_dir = Path(output) / split / category
            for video_path in tqdm(video_dir.iterdir(), desc=f'Saving {split}/{category} frames'):
                if not video_path.suffix == '.mp4':
                    continue
                save_path = output_dir / video_path.stem
                Path.mkdir(save_path, parents=True)
                extract_and_save_key_frames(video_path, save_path, num_keyframes=15)

            model_path = Path(model)
            for frame_dir in output_dir.iterdir():
                sorted_frame_list = sorted(frame_dir.iterdir(), key=lambda x: int(x.stem.split('_')[1]))
                extract_keypoints_with_pose_model(frame_paths=sorted_frame_list,
                                                  model_path=model_path,
                                                  output_csv_path=frame_dir)


@tcn.command(name='train')
@click.option('--dataset', type=Path, required=True, help='Path to extracted dataset directory containing "train" and "val" subdirectories.')
@click.option('--epochs', type=int, required=True, help='Number of epochs to train the model.')
@click.option('--batch_size', type=int, required=False, default=2, help='Number of batches to load the dataset.')
@click.option('--lr', type=float, required=False, default=0.001, help='learning rate')
@click.option('--patience', type=int, required=False, default=10, help=' Early stopping patience.')
@click.option('--save_path', type=Path, required=True, help='Path to save best model and training plots.')
def tcn_train(dataset, epochs, batch_size, lr, patience, save_path):
    train(dataset, epochs, batch_size, lr, patience, save_path)


@tcn.command(name='evaluate')
@click.option('--model', type=Path, required=True, help='Path to model for evaluation.')
@click.option('--data', type=Path, required=True, help='Path to dataset directory for evaluation.')
@click.option('--batch_size', type=int, required=False, default=2, help='Batch size to load data for evaluation')
@click.option('--split', type=str, required=False, default='val', help='"test": to evaluate test dataset. "val": to evaluate validation dataset.')
def tcn_evaluate(model, data, batch_size, split):
    eval_metrics = evaluate(model, data, batch_size, split)
    print(f"{split} Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"{split} Precision: {eval_metrics['precision']:.4f}")
    print(f"{split} Recall: {eval_metrics['recall']:.4f}")
    print(f"{split} F1: {eval_metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{eval_metrics['confusion_matrix']}")


if __name__ == '__main__':
    tcn()
