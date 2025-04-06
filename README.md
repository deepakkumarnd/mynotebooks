# setup venv 

pip3 install virtualenv
python3 -m venv <env-name>
source <env-namei>/bin/activate


# install spacy

pip3 install -U pip setuptools wheel
pip3 install -U 'spacy[apple]'
python3 -m spacy download en_core_web_sm


# jupyter notebook installation

pip3 install notebook
source <env-namei>/bin/activate
jupyter notebook


