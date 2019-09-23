# Prerequisites

1. Download Town07 from [CARLA release page](https://github.com/carla-simulator/carla/releases/tag/0.9.6) and extract the content to your carla root folder

2. Install the following python packages into your virtual environment:
```bash
pip install py_trees==0.8.3 networkx==2.2 psutil shapely xmlschema opencv-python
```

3. If not yet done, add the carla and agent modules to your PYTHONPATH:

```bash
export CARLA_ROOT=path/to/carla/root_folder/
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla
```

# How To Evaluate Your Agent

**Note**: This introduction works for an agent that uses only one camera for action inference. For other settings, you need to modify
[PrakAgent.py](https://ids-git.fzi.de/svmuelle/carla_rllib/blob/prak_evaluator/carla_rllib/prak_evaluator/prak_evaluation/PrakAgent.py) and
[agent_config.py](https://ids-git.fzi.de/svmuelle/carla_rllib/blob/prak_evaluator/carla_rllib/prak_evaluator/prak_evaluation/agent_config.json) accordingly.

1. Adjust the agent configuration file ([agent_config.py](https://ids-git.fzi.de/svmuelle/carla_rllib/blob/prak_evaluator/carla_rllib/prak_evaluator/prak_evaluation/agent_config.json))
    - camera: specify your camera settings (type, position, image size and field of view)
    - policy_checkpoint: path to policy weights (not mandatory but might be used to load weights in PrakAgent.py)
    - debug: specify if you want to debug the evaluation (debug mode allows to validate the scenarios or camera settings)
2. Adjust the agent class ([PrakAgent.py](https://ids-git.fzi.de/svmuelle/carla_rllib/blob/prak_evaluator/carla_rllib/prak_evaluator/prak_evaluation/PrakAgent.py))
    - Line 65: build (load) your policy (weights)
    - Line 128: implement the action inference
3. Run ```./run_evaluation.sh ```