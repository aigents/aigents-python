# aigents-python
Python API to Aigents backend/core and experienttial learning experiments

## Setting up Jupyter on remote Ubuntu server in the cloud and run it locally

### Do this on remote server in the cloud (or if you want to access Jupyter locally):

#### Using Python 3.11.13

1. Make sure you have Python3 pip and vitualenv installed, see https://1cloud.ru/help/linux/ustanovka-jupyter-notebook-na-ubuntu-18-04 *(do this only once)*
1. git clone https://github.com/aigents/aigents-python.git *(do this only once)*
1. cd aigents-python
1. virtualenv env *or* python -m venv env *(do this only once)*
1. . env/bin/activate *(. ./env/Scripts/activate [- if under Windows](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html))*
1. pip install -r requirements.txt *(do this only once)*
1. sudo iptables -A INPUT -p tcp --dport 8888 -j ACCEPT *(do this only once, need only to deploy on cloud server)*
1. jupyter notebook --no-browser --port=8888 *(if you are doing this on your local machine, just access http://localhost:8888/ locally in the browser)*

### Do this on your local machine (if you want to access remote Jupyter):

1. `ssh -i <yourkey>.pem -N -f -L localhost:9999:localhost:8888 <yourusername>@<yourhost>` (do this in terminal)
1. http://localhost:9999/ (access this in the browser)

## Contents

1. [Aigents Backend API and Tools](https://github.com/aigents/aigents-python/tree/main/aigents-python)
1. [Experiential Learning with Open AI Gym](https://github.com/aigents/aigents-python/tree/main/aigents-gym)
     
