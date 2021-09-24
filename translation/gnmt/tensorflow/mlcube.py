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


class RunPerformanceTask(object):
    """Run performance task Class
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

        
        env = os.environ.copy()
        env.update({
            'DATA_DIR': os.path.join(data_dir, 'nmt', 'data'),
            'MODEL_DIR': os.path.join(model_dir, 'ende_gnmt_model_4_layer'),
            'OUTPUT_DIR': output_dir,
            'TASK': "performance"
        })

        env.update(parameters)

        process = subprocess.Popen("./run.sh", cwd=".", env=env)
        process.wait()


class RunAccuracyTask(object):
    """Run performance task Class
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

        
        env = os.environ.copy()
        env.update({
            'DATA_DIR': os.path.join(data_dir, 'nmt', 'data'),
            'MODEL_DIR': os.path.join(model_dir, 'ende_gnmt_model_4_layer'),
            'OUTPUT_DIR': output_dir,
            'TASK': "accuracy"
        })

        env.update(parameters)

        process = subprocess.Popen("./run.sh", cwd=".", env=env)
        process.wait()


@app.command("download_data")
def download_data(data_dir: str = typer.Option(..., "--data_dir")):
    DownloadDataTask.run(data_dir)


@app.command("download_model")
def download_model(model_dir: str = typer.Option(..., "--model_dir")):
    DownloadModelTask.run(model_dir)


@app.command("run_performance")
def run_performance(
    data_dir: str = typer.Option(..., "--data_dir"),
    model_dir: str = typer.Option(..., "--model_dir"),
    output_dir: str = typer.Option(..., "--output_dir"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
):
    RunPerformanceTask.run(data_dir, model_dir, output_dir, parameters_file)

@app.command("run_accuracy")
def run_accuracy(
    data_dir: str = typer.Option(..., "--data_dir"),
    model_dir: str = typer.Option(..., "--model_dir"),
    output_dir: str = typer.Option(..., "--output_dir"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
):
    RunAccuracyTask.run(data_dir, model_dir, output_dir, parameters_file)


if __name__ == "__main__":
    app()
