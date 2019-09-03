# Prerequisites

##Virtual env erzeugen:

virtualenv -p python3 carla_env
source carla_env/bin/activate

##Bashrc anpassen:
CARLA_ROOT="/fzi/ids/rottach/no_backup/carla_0.9.6"
export CARLA_ROOT
alias sbas="source ~/.bashrc"
alias ebas="vim ~/.bashrc"
alias source_carla="source ~/no_backup/carla_env/bin/activate"

# Installation CARLA 0.9.6 

0. get CARLA 0.9.6
	copy precompiled version from sven mueller
	
	$ cd /disk/vanishing_data/svmuelle/
	$ cp carla-nopng_0.9.6.tar.gz ~/no_backup/
	$ cd ~/no_backup/
	$ tar -zxvf carla-nopng_0.9.6.tar.gz
	
	edit CARLA_ROOT in bashrc(add following lines if necessary):

	pythonpath necessary to point python to carla_rllib (maybe not necessary in future versions)

	export CARLA_ROOT=/fzi/ids/zanger/no_backup/zanger/carla_0.9.6/
	export PYTHONPATH="${PYTHONPATH}:/fzi/ids/zanger/no_backup/carla_rllib/"
	export PATH=$PATH:$CARLA_ROOT

1. Clone wrapper repo

	$ git clone https://ids-git.fzi.de/svmuelle/carla_rllib.git
2. set up virtual environment, Python should be 3.5.0
3. checkout branch develop
	$ git checkout develop
4. Install the required python packages (into your virtual environment):

	$ cd carla_rllib
	$ pip install .
5. run test files and probably have errors:
	$ python carla_env_test.py
6. Check Versions, python, carla client and carla server have to be the same version
	if carla pip installs are existent, check versions
	$ pip list --local
7. remove pip version if not 0.9.6 and use egg files to reinstall carla 
	$ pip uninstall carla
	$ cd /disk/no_backup/zanger/carla_0.9.6/PythonAPI/carla/dist/
	$ ls
	
	egg files should show up, install with easyinstall
		
	$ easy_install carla-0.9.6-py3.5-linux-x86_64.egg
8. run CARLA_UE4.sh (or SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 sh CarlaUE4.sh )
9. run wrapper test files (somewhere in carla_rllib/examples)
	$ python carla_env_test.py 