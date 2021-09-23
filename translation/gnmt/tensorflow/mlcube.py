"""MLCube handler file"""
import os
import yaml
import typer
import shutil
import subprocess
from pathlib import Path


app = typer.Typer()


class DownloadDataTask(object):
    """Download data task Class
    It defines the environment variables:
        DATA_DIR: Directory path to download the dataset
    Then executes the download script"""

    @staticmethod
    def run(data_dir: str) -> None:

        env = os.environ.copy()
        env.update(
            {"DATA_DIR": data_dir,}
        )

        process = subprocess.Popen("./download_dataset.sh", cwd=".", env=env)
        process.wait()
        # Verify dataset after download
        process = subprocess.Popen("./verify_dataset.sh", cwd=".", env=env)
        process.wait()


class DownloadModelTask(object):
    """Download model task Class
    It defines the environment variables:
        MODEL_DIR: Directory path to download the model
    Then executes the download script"""

    @staticmethod
    def run(model_dir: str) -> None:

        env = os.environ.copy()
        env.update(
            {"MODEL_DIR": model_dir,}
        )

        process = subprocess.Popen("./download_trained_model.sh", cwd=".", env=env)
        process.wait()


class RunInferenceTask(object):
    """Run inference task Class
    It defines the environment variables:
        DATA_DIR: Dataset directory path
        MODEL_DIR: Model directory path
        OUTPUT_DIR: Directory path where model will be saved
        All other parameters defined in parameters_file
    Then executes the benchmark script"""

    @staticmethod
    def run(data_dir: str, model_dir: str, output_dir: str, parameters_file: str) -> None:
        with open(parameters_file, "r") as stream:
            parameters = yaml.safe_load(stream)

        command = f"python run_task.py --run=performance --batch_size={parameters['batch_size']} --dataset_path={data_dir} --model_path={model_dir} --output_path={output_dir}"
        splitted_command = command.split()
        process = subprocess.Popen(splitted_command, cwd=".")
        process.wait()


@app.command("download_data")
def download_data(data_dir: str = typer.Option(..., "--data_dir")):
    DownloadDataTask.run(data_dir)


@app.command("download_model")
def download_model(model_dir: str = typer.Option(..., "--model_dir")):
    DownloadModelTask.run(model_dir)


@app.command("run_inference")
def run_inference(
    data_dir: str = typer.Option(..., "--data_dir"),
    model_dir: str = typer.Option(..., "--model_dir"),
    output_dir: str = typer.Option(..., "--output_dir"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
):
    RunInferenceTask.run(data_dir, model_dir, output_dir, parameters_file)


if __name__ == "__main__":
    app()
