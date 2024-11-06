# STSA - Capturing Spatio-Temporal Dependencies with Competitive Set Attention for Video Summarization  


The Official Github Repository of "Capturing Spatio-Temporal Dependencies with Competitive Set Attention for Video Summarization". 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14032949.svg)](https://doi.org/10.5281/zenodo.14032949)

# Download the datasets from the link below

https://drive.google.com/drive/folders/1KTpftiMchP0q-pdcJ4K7HaARJloOwQA6?usp=sharing

Create a datasets folder inside the ./STSA/datasets folder and save the downloaded .h5 files of the datasets.

# Create STSA conda environment

```conda create -n STSA

# Activate STSA environment
'''conda activate STSA'''

# Install the packages provided in the requirement.txt
'''pip install -r requirements.txt'''

# To train the model
'''python3 train.py --exp_name 'ExperimentName' --dataset 'TVSum or SumMe' --batch_size #BATCH_SIZE --epochs #EPOCHS'''

The evaluation is provided in the training file.

# Requiremnts

Package                       Version
----------------------------- --------------------
absl-py                       2.1.0
actionlib                     1.14.0
aiofiles                      22.1.0
aiohttp                       3.8.4
aiosignal                     1.3.1
aiosqlite                     0.20.0
alabaster                     0.7.13
albumentations                1.0.3
aliyun-python-sdk-core        2.15.1
aliyun-python-sdk-kms         2.16.2
angles                        1.9.13
anyio                         4.3.0
appdirs                       1.4.4
apturl                        0.5.2
argon2-cffi                   23.1.0
argon2-cffi-bindings          21.2.0
arrow                         1.3.0
asn1crypto                    1.2.0
astor                         0.8.1
asttokens                     2.4.1
astunparse                    1.6.3
async-timeout                 4.0.2
attrs                         23.2.0
autobahn                      17.10.1
Automat                       0.8.0
Babel                         2.14.0
backcall                      0.2.0
bagpy                         0.4.10
bcrypt                        3.1.7
beautifulsoup4                4.12.3
bitarray                      2.9.2
bitstring                     4.1.4
bleach                        6.1.0
blinker                       1.4
Brlapi                        0.7.0
cachetools                    4.0.0
camera-calibration-parsers    1.12.0
catkin                        0.8.10
catkin-pkg                    0.4.23
catkin-pkg-modules            1.0.0
cbor                          1.0.0
certifi                       2019.11.28
cffi                          1.13.2
chardet                       3.0.4
charset-normalizer            2.1.1
click                         8.1.7
cloudpickle                   1.6.0
colorama                      0.4.6
comm                          0.2.1
command-not-found             0.3
commonmark                    0.9.1
constantly                    15.1.0
contourpy                     1.0.7
crcmod                        1.7
cryptography                  2.8
cupshelpers                   1.0
cv-bridge                     1.16.2
cycler                        0.10.0
Cython                        0.29.14
datasets                      3.0.1
dbus-python                   1.2.16
debugpy                       1.8.1
decorator                     5.1.1
defer                         1.0.6
defusedxml                    0.6.0
diagnostic-updater            1.11.0
dill                          0.3.8
distlib                       0.3.8
distro                        1.4.0
distro-info                   0.23ubuntu1
docutils                      0.16
duplicity                     0.8.12.0
dynamic-reconfigure           1.7.3
einops                        0.8.0
empy                          3.3.2
entrypoints                   0.3
environs                      8.0.0
et-xmlfile                    1.1.0
exceptiongroup                1.2.0
executing                     2.0.1
fasteners                     0.14.1
fastjsonschema                2.19.1
filelock                      3.13.1
flatbuffers                   24.3.25
fonttools                     4.39.3
fqdn                          1.5.1
frozenlist                    1.3.3
fsspec                        2023.5.0
future                        0.18.2
gast                          0.3.3
gencpp                        0.7.0
geneus                        3.0.0
genlisp                       0.4.18
genmsg                        0.6.0
gennodejs                     2.0.2
genpy                         0.6.15
ghp-import                    2.1.0
gnupg                         2.3.1
google-auth                   1.35.0
google-auth-oauthlib          0.4.6
google-pasta                  0.2.0
grpcio                        1.32.0
gym                           0.17.3
h5py                          2.10.0
httplib2                      0.14.0
huggingface-hub               0.23.4
hyperlink                     19.0.0
idna                          2.8
image-geometry                1.16.2
imageio                       2.34.0
imagesize                     1.4.1
importlib-metadata            7.0.1
importlib-resources           5.12.0
incremental                   16.10.1
inplace-abn                   1.1.0
interactive-markers           1.12.0
ipykernel                     6.29.3
ipython                       8.12.3
ipython-genutils              0.2.0
ipywidgets                    8.1.5
isoduration                   20.11.0
jderobot-jderobottypes        1.0.0
jedi                          0.19.1
Jinja2                        3.1.3
jmespath                      0.10.0
joblib                        1.3.2
json5                         0.9.17
jsonpointer                   3.0.0
jsonschema                    4.21.1
jsonschema-specifications     2023.12.1
jupyter                       1.1.1
jupyter_client                7.4.9
jupyter-console               6.6.3
jupyter_core                  5.7.1
jupyter-events                0.9.0
jupyter_server                2.12.5
jupyter_server_fileid         0.9.1
jupyter_server_terminals      0.5.2
jupyter_server_ydoc           0.8.0
jupyter-ydoc                  0.2.5
jupyterlab                    3.6.7
jupyterlab_pygments           0.3.0
jupyterlab_server             2.25.3
jupyterlab_widgets            3.0.13
kagglehub                     0.2.9
keras                         2.11.0
Keras-Applications            1.0.8
Keras-Preprocessing           1.1.2
keyboard                      0.13.5
keyring                       18.0.1
kiwisolver                    1.2.0
language-selector             0.1
laser_geometry                1.6.7
launchpadlib                  1.10.13
lazr.restfulclient            0.14.2
lazr.uri                      1.0.3
libclang                      16.0.6
lockfile                      0.12.2
louis                         3.12.0
lz4                           3.0.2+dfsg
m2r2                          0.3.3.post2
macaroonbakery                1.3.1
Mako                          1.1.0
Markdown                      3.1.1
markdown-it-py                3.0.0
MarkupSafe                    2.1.2
marshmallow                   3.21.0
matplotlib                    3.1.2
matplotlib-inline             0.1.6
mdurl                         0.1.2
mergedeep                     1.3.4
message-filters               1.16.0
mistune                       3.0.2
mkdocs                        1.5.3
model-index                   0.1.11
monotonic                     1.5
more-itertools                8.2.0
mpi4py                        3.0.3
multidict                     6.0.4
multiprocess                  0.70.16
natsort                       7.0.1
nbclassic                     1.0.0
nbclient                      0.9.0
nbconvert                     7.16.1
nbformat                      5.9.2
nest-asyncio                  1.6.0
netifaces                     0.10.9
networkx                      3.1
nose                          1.3.7
notebook                      6.5.6
notebook_shim                 0.2.4
npyscreen                     4.10.5
numpy                         1.20.3
nvidia-cublas-cu11            11.10.3.66
nvidia-cuda-nvrtc-cu11        11.7.99
nvidia-cuda-runtime-cu11      11.7.99
nvidia-cudnn-cu11             8.5.0.96
nvidia-ml-py                  11.525.131
nvitop                        1.1.2
oauthlib                      3.1.0
olefile                       0.46
opencv-python-headless        4.9.0.80
opendatalab                   0.0.10
openmim                       0.3.9
openpyxl                      3.1.4
openxlab                      0.0.38
opt-einsum                    3.3.0
ordered-set                   4.1.0
ortools                       7.8.7959
oss2                          2.17.0
overrides                     7.7.0
packaging                     24.1
pandas                        2.0.3
pandocfilters                 1.5.1
paramiko                      2.6.0
parso                         0.8.3
pathlib                       1.0.1
pathspec                      0.12.1
pexpect                       4.6.0
pickleshare                   0.7.5
pillow                        10.2.0
pip                           24.2
pkgutil_resolve_name          1.3.10
platformdirs                  4.1.0
pluggy                        0.13.1
portalocker                   1.5.2
prometheus_client             0.20.0
prompt-toolkit                3.0.43
protobuf                      5.28.3
psutil                        5.9.5
ptlflow                       0.2.7
ptyprocess                    0.7.0
pure-eval                     0.2.2
py                            1.8.1
py-ubjson                     0.14.0
py3rosmsgs                    1.18.2
pyarrow                       17.0.0
pyasn1                        0.4.8
pyasn1-modules                0.2.8
pycairo                       1.16.2
pycocotools                   2.0.6
pycparser                     2.19
pycryptodome                  3.20.0
pycryptodomex                 3.19.1
pycups                        1.9.73
pydantic                      1.10.2
pydash                        7.0.7
pyDeprecate                   0.3.2
pydot                         1.4.1
pygame                        2.5.2
pyglet                        1.5.0
Pygments                      2.17.2
PyGObject                     3.36.0
PyHamcrest                    1.9.0
PyJWT                         1.7.1
pymacaroons                   0.13.0
pymdown-extensions            10.7
PyNaCl                        1.3.0
pynput                        1.6.8
PyOpenGL                      3.1.0
pyOpenSSL                     19.1.0
pyparsing                     2.4.6
pypng                         0.0.21
PyQRCode                      1.2.1
PyQt3D                        5.15.0
PyQt5                         5.15.0
PyQt5-sip                     12.8.1
pyRFC3339                     1.1
pyserial                      3.5
PySocks                       1.7.1
pytest                        5.4.0
python-apt                    2.0.0+ubuntu0.20.4.8
python-dateutil               2.8.2
python-debian                 0.1.36ubuntu1
python-dotenv                 1.0.1
python-gnupg                  0.4.5
python-json-logger            2.0.7
python-qt-binding             0.4.4
python-snappy                 0.5.3
python-xlib                   0.33
pytorch-lightning             1.5.10
pytorch-model-summary         0.1.2
PyTrie                        0.2
pytz                          2023.4
PyWavelets                    1.4.1
pyxdg                         0.26
PyYAML                        6.0.1
pyyaml_env_tag                0.1
pyzmq                         24.0.1
qt-dotgraph                   0.4.2
qt-gui                        0.4.2
qt-gui-py-common              0.4.2
randaugment                   1.0.2
recommonmark                  0.7.1
referencing                   0.33.0
regex                         2023.12.25
reportlab                     3.5.34
requests                      2.32.3
requests-oauthlib             1.3.0
requests-unixsocket           0.2.0
resource_retriever            1.12.7
retrying                      1.3.3
rfc3339-validator             0.1.4
rfc3986-validator             0.1.1
rich                          13.4.2
rinoh-typeface-dejavuserif    0.1.3
rinoh-typeface-texgyrecursor  0.1.1
rinoh-typeface-texgyreheros   0.1.1
rinoh-typeface-texgyrepagella 0.1.1
rinohtype                     0.5.4
roman                         2.0.0
rosbag                        1.16.0
rosboost-cfg                  1.15.8
rosclean                      1.15.8
rosdep-modules                0.22.2
rosdistro                     0.9.0
rosdistro-modules             0.9.0
rosgraph                      1.16.0
roslaunch                     1.16.0
roslib                        1.15.8
roslint                       0.12.0
roslz4                        1.16.0
rosmake                       1.15.8
rosmaster                     1.16.0
rosmsg                        1.16.0
rosnode                       1.16.0
rosparam                      1.16.0
rospkg                        1.2.9
rospkg-modules                1.5.0
rospy                         1.16.0
rosservice                    1.16.0
rostest                       1.16.0
rostopic                      1.16.0
rosunit                       1.15.8
roswtf                        1.16.0
rpds-py                       0.18.0
rqt_gui                       0.5.3
rqt_gui_py                    0.5.3
rsa                           4.0
ruamel.yaml                   0.15.87
rviz                          1.14.20
sacremoses                    0.1.1
safetensors                   0.4.3
scikit-image                  0.19.2
scikit-learn                  1.0.2
scipy                         1.4.1
seaborn                       0.12.2
SecretStorage                 2.3.1
Send2Trash                    1.8.2
sensor-msgs                   1.13.1
sentencepiece                 0.2.0
service-identity              18.1.0
setuptools                    60.2.0
simplejson                    3.16.0
sip                           4.19.21
six                           1.15.0
smach                         2.5.2
smclib                        1.8.6
sniffio                       1.3.1
snowballstemmer               2.2.0
soupsieve                     2.5
Sphinx                        3.2.1
sphinx-autodoc-typehints      1.4.0
sphinx-bootstrap-theme        0.8.1
sphinx-markdown-parser        0.2.4
sphinx-rtd-theme              2.0.0
sphinxcontrib-applehelp       1.0.4
sphinxcontrib-devhelp         1.0.2
sphinxcontrib-htmlhelp        2.0.1
sphinxcontrib-jquery          4.1
sphinxcontrib-jsmath          1.0.1
sphinxcontrib-qthelp          1.0.3
sphinxcontrib-serializinghtml 1.1.5
split-folders                 0.5.1
ssh-import-id                 5.10
stack-data                    0.6.3
systemd-python                234
tabulate                      0.8.10
tenacity                      8.2.2
tensorboard                   2.2.2
tensorboard-data-server       0.6.1
tensorboard-plugin-wit        1.8.0
tensorboardX                  2.6.2.2
tensorflow                    2.4.0
tensorflow-estimator          2.2.0
tensorflow-gpu                2.2.0
tensorflow-io-gcs-filesystem  0.34.0
termcolor                     1.1.0
terminado                     0.18.0
tf                            1.13.2
tf2-geometry-msgs             0.7.7
tf2-py                        0.7.7
tf2-ros                       0.7.7
threadpoolctl                 3.3.0
tifffile                      2023.7.10
timm                          1.0.8
tinycss2                      1.2.1
tokenizers                    0.10.3
tomli                         2.0.1
topic-tools                   1.16.0
torch                         1.10.1+cu111
torchaudio                    0.10.1+cu111
torchmetrics                  0.8.2
torchvision                   0.11.2+cu111
tornado                       6.4
tqdm                          4.66.5
traitlets                     5.14.1
transformers                  4.13.0
transforms3d                  0.4.1
Twisted                       18.9.0
txaio                         2.10.0
types-python-dateutil         2.9.0.20240821
typing                        3.7.4.1
typing_extensions             4.12.2
tzdata                        2023.4
u-msgpack-python              2.1
ubuntu-advantage-tools        27.10
ubuntu-drivers-common         0.0.0
ufw                           0.36
unattended-upgrades           0.1
unify                         0.5
untokenize                    0.1.1
uri-template                  1.3.0
urllib3                       1.25.8
usb-creator                   0.3.7
vcstool                       0.2.14
virtualenv                    20.25.0
wadllib                       1.3.3
watchdog                      4.0.0
wcwidth                       0.1.8
webcolors                     24.8.0
webencodings                  0.5.1
websocket-client              1.7.0
Werkzeug                      3.0.1
wheel                         0.43.0
widgetsnbextension            4.0.13
wrapt                         1.12.1
wsaccel                       0.6.2
xkit                          0.0.0
xmltodict                     0.12.0
xxhash                        3.5.0
y-py                          0.6.2
yapf                          0.40.2
yarl                          1.9.2
ypy-websocket                 0.8.4
ytsphinx                      1.2.1.dev20200430
zipp                          3.1.0
zope.interface                4.7.1

## Acknowledgments

- **[Bin SHENG and ]** - Provided insights and guidance on the project. Find more about his work on [https://scholar.google.com/citations?user=QlGJBvkAAAAJ&hl=en].

