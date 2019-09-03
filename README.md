################## Starten
Erstmal ein virtual env erzeugen: 

virtualenv -p python3 carla_env
source carla_env/bin/activate
pip install pygame numpy networkx


- Zun?chst brauchen wir diese Umgebungsvariable: 
export CARLA_ROOT=/carla-nopng-0.9.5/ \n
export PATH=$PATH:$CARLA_ROOT

- Carla liegt unter root/carla-nopng-0.9.5/ (z.B. rottach@ids-hubbard:/carla-nopng-0.9.5$)
- CarlaUE4.sh startet Carla ohne Parameter
- ./CarlaUE4.sh /HDMaps/Town02 startet die zweite Stadt
- Alle Parameter wie z.B. Auto usw. sind in CarlaSettings.ini welches als parameter ?bergeben wird:
also z.B. -carla-settings=CarlaSettings.ini


Sieht beispielsweise so aus:
NumberOfVehicles=60
NumberOfPedestrians=60
WeatherId=3


WICHTIG:
1. Starte CarlaUE4.sh -carla-server (Server)
(1. OHNE GUI: SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 sh CarlaUE4.sh -carla-server)
2. Starte die python api (Client) z.B. mit pygame

################ Installation CARLA 0.9.6 

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
8. run CARLA_UE4.sh
9. run wrapper test files (somewhere in carla_rllib/examples)
	$ python carla_env_test.py 


################ Optimierung
cd /
cp -r carla-nopng-0.9.5/ ~/no_backup/

cd ~/no_backup/carla-nopng-0.9.5/CarlaUE4/Config/
kate DefaultEngine.ini
--> r.TextureStreaming=True

################# Schlie?en
- Manuell mit ALT + SPACE und dann auf exit dr?cken


################ Running macad gym wrapper
source carla_env/bin/activate
python macad-gym/setup.py build
python macad-gym/setup.py install
export CARLA_ROOT=/carla-nopng-0.9.5/
export PATH=$PATH:$CARLA_ROOT
export CARLA_SERVER=/carla-nopng-0.9.5/CarlaUE4.sh

dann:
import macad_gym
env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")

ggf. macad_gym.src.macad_gym
