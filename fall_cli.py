from pathlib import Path
import click
from main_inference import main_inference


@click.group()
def fallDetection():
    """ Group for Fall Detection commands."""
    pass

@fallDetection.command(name='inference')
@click.option('--source', type=str, required=True, help='Path to the video file or camera Index.')
@click.option('--pose_model', type=Path, required=True, help='Path to the fine-tuned pose estimation model.')
@click.option('--tcn_model', type=Path, required=True, help='Path to the trained TCN model.')
@click.option('--save_dir', type=Path, required=False, default=None, help='Path to the directory for saving results, if provided.')
@click.option('--show_video', type=bool, required=False, default=False, help='Whether to display the video stream (default: False).')
def fall_detection_inference(source, pose_model, tcn_model, save_dir, show_video):
    main_inference(source, pose_model, tcn_model, save_dir, show_video)


if __name__ == '__main__':
    fallDetection()
