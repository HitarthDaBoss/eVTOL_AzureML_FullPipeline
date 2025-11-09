from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget
import os

# ensure you created config.json or authenticated
ws = Workspace.from_config()
compute_name = os.environ.get("AML_COMPUTE_NAME", "gpu-cluster")
compute_target = ComputeTarget(workspace=ws, name=compute_name)

env = Environment.from_conda_specification(name="evtol-azure-env", file_path="environment.yml")

src = ScriptRunConfig(
    source_directory="src",
    script="train_rl.py",
    arguments=["--cfg", "../configs/default.yaml"],
    compute_target=compute_target,
    environment=env
)

experiment = Experiment(ws, "eVTOL-PPO")
run = experiment.submit(src)
print("Submitted run", run.id)
run.wait_for_completion(show_output=True)
