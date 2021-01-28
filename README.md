# How to use the repository
### Models
- The models can be found under ddpg/CARLA or stable_baselines/carla_env_MIF or A3C_continuous
- For the stable baselines models use the parameters MODE and SMODE to define the model and the running mode
- For the DDPG use the run.py for configuration and start up
- Models can also be reloaded in the stable baselines file, we pushed some of the models - the most recent ones perform the best
- We have also pushed a custom PPO implementation
- to switch input states, modify flags   

		self._data_gen  

        self._use_front_ae  

        self._use_birdseye  

	in the carla_env script (warning: switch training model to CNN/MLP policy in carla_env_MIF to fit the input)
### Birds-eye-view
- The implementation of the encoder-decoder architecture can be found under semantic_birdseyeview
- There are also some models pushed
- separate requirements.txt in semantic_birdseyeview
- use the script train_model_rgb to train the birds-eye-view based on rgb images (should located in a "camera_storage" folder as npy)
- use the script train_model_semantic to train the birds-eye-view based on segmented images (should located in a "camera_storage" folder as npy)

### Wrapper
- We used the carla_rllib wrapper from Sven Müller
- We have done some adjustments for the terminal condition and the sensor types under carlarllib/wrapper/carla_wrapper.py
- The observation space and reward function are modified in carlarllib/environment/carla_environment/carla_environment.py



# Prerequisites
```console
virtualenv -p python3 carla_env  

source carla_env/bin/activate  

pip install -r requirements.txt  
```

## Bashrc:
```console
CARLA_ROOT="<dir>/carla_0.9.6"  
export CARLA_ROOT  
``` 

## Installation CARLA 0.9.6 
0. get CARLA 0.9.6  
	download precompiled version
	
	edit CARLA_ROOT in bashrc(add following lines if necessary):  
	(pythonpath necessary to point python to carla_rllib (maybe not necessary in future versions))  
	
	```console
	export CARLA_ROOT=<dir>/carla_0.9.6/ 
	export PYTHONPATH="${PYTHONPATH}:<dir>/carla_rllib/"  
	export PATH=$PATH:$CARLA_ROOT 
	```

1. Clone wrapper repo  
	```console
	git clone https://ids-git.fzi.de/svmuelle/carla_rllib.git
	```
2. set up virtual environment, Python should be 3.5.0
3. checkout branch develop  
	```console
	git checkout develop
	```
4. Install the required python packages into your virtual environment:  
	```console
	cd carla_rllib  
	pip install .  
	```
5. run test files:
	```console
	python carla_env_test.py
	```
6. If you have errors, check Versions, python, carla client and carla server have to be the same version
	if carla pip installs are existent, check versions
	```console
	pip list --local  
	```
7. remove pip version if not 0.9.6 and use egg files to reinstall carla  
	```console
	pip uninstall carla
	cd <dir>/carla_0.9.6/PythonAPI/carla/dist/  
	easy_install carla-0.9.6-py3.5-linux-x86_64.egg  
	```
8. run CARLA_UE4.sh (or SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 sh CarlaUE4.sh )  
9. run wrapper test files (in carla_rllib/examples)  
	```console
	python carla_env_test.py
	```
