# STSA - Capturing Spatio-Temporal Dependencies with Competitive Set Attention for Video Summarization  

[![DOI](https://zenodo.org/badge/882984239.svg)](https://doi.org/10.5281/zenodo.14032948)

The Official Github Repository of "Capturing Spatio-Temporal Dependencies with Competitive Set Attention for Video Summarization". 

This paper has been accepted in The Visual Computer [![paper](https://doi.org/10.1007/s00371-025-03865-1)](https://doi.org/10.1007/s00371-025-03865-1).

# Download the datasets from the link below

https://drive.google.com/drive/folders/1KTpftiMchP0q-pdcJ4K7HaARJloOwQA6?usp=sharing

Access will be provided based on access requests.

Create a datasets folder inside the ./STSA/datasets folder and save the downloaded .h5 files of the datasets.

# Create STSA conda environment

```
conda create -n STSA
```

# Activate STSA environment
```
conda activate STSA
```

# Install the packages provided in the requirement.txt
```
pip install -r requirements.txt
```

# To train the model
```
python3 train.py --exp_name 'ExperimentName' --dataset 'DatasetName(TVSum or SumMe)' --batch_size #BATCH_SIZE --epochs #EPOCHS
```

# Evaluation

The evaluation is provided in the training file.

# Requirements
The mandate requirement packages:
```
torch                
torchaudio           
torchsummary 
torchvision
torchviz
 ```



| Package                      | Version       |
|------------------------------|---------------|
| datasets                     | 3.0.1         |
| distlib                      | 0.3.8         |
| einops                       | 0.8.0         |
| environs                     | 8.0.0         |
| et-xmlfile                   | 1.1.0         |
| filelock                     | 3.13.1        |
| fonttools                    | 4.39.3        |
| fqdn                         | 1.5.1         |
| google-auth                  | 1.35.0        |
| google-auth-oauthlib         | 0.4.6         |
| google-pasta                 | 0.2.0         |
| gym                          | 0.17.3        |
| h5py                         | 2.10.0        |
| httplib2                     | 0.14.0        |
| huggingface-hub              | 0.23.4        |
| hyperlink                    | 19.0.0        |
| image-geometry               | 1.16.2        |
| imageio                      | 2.34.0        |
| imagesize                    | 1.4.1         |
| importlib-metadata           | 7.0.1         |
| importlib-resources          | 5.12.0        |
| incremental                  | 16.10.1       |
| inplace-abn                  | 1.1.0         |
| interactive-markers          | 1.12.0        |
| ipykernel                    | 6.29.3        |
| ipython                      | 8.12.3        |
| ipython-genutils             | 0.2.0         |
| ipywidgets                   | 8.1.5         |
| isoduration                  | 20.11.0       |
| jderobot-jderobottypes       | 1.0.0         |
| jedi                         | 0.19.1        |
| Jinja2                       | 3.1.3         |
| jmespath                     | 0.10.0        |
| joblib                       | 1.3.2         |
| json5                        | 0.9.17        |
| jsonpointer                  | 3.0.0         |
| jsonschema                   | 4.21.1        |
| jsonschema-specifications    | 2023.12.1     |
| jupyter                      | 1.1.1         |
| jupyter_client               | 7.4.9         |
| jupyter-console              | 6.6.3         |
| jupyter_core                 | 5.7.1         |
| jupyter-events               | 0.9.0         |
| jupyter_server               | 2.12.5        |
| jupyter_server_fileid        | 0.9.1         |
| jupyter_server_terminals     | 0.5.2         |
| jupyter_server_ydoc          | 0.8.0         |
| jupyter-ydoc                 | 0.2.5         |
| jupyterlab                   | 3.6.7         |
| jupyterlab_pygments          | 0.3.0         |
| jupyterlab_server            | 2.25.3        |
| jupyterlab_widgets           | 3.0.13        |
| kagglehub                    | 0.2.9         |
| keras                        | 2.11.0        |
| Keras-Applications           | 1.0.8         |
| Keras-Preprocessing          | 1.1.2         |
| keyboard                     | 0.13.5        |
| keyring                      | 18.0.1        |
| kiwisolver                   | 1.2.0         |
| Markdown                     | 3.1.1         |
| markdown-it-py               | 3.0.0         |
| matplotlib                   | 3.1.2         |
| matplotlib-inline            | 0.1.6         |
| multidict                    | 6.0.4         |
| multiprocess                 | 0.70.16       |
| notebook                     | 6.5.6         |
| notebook_shim                | 0.2.4         |
| npyscreen                    | 4.10.5        |
| numpy                        | 1.20.3        |
| nvidia-cublas-cu11           | 11.10.3.66    |
| nvidia-cuda-nvrtc-cu11       | 11.7.99       |
| nvidia-cuda-runtime-cu11     | 11.7.99       |
| nvidia-cudnn-cu11            | 8.5.0.96      |
| nvidia-ml-py                 | 11.525.131    |
| nvitop                       | 1.1.2         |
| opencv-python-headless       | 4.9.0.80      |
| ortools                      | 7.8.7959      |
| oss2                         | 2.17.0        |
| packaging                    | 24.1          |
| pandas                       | 2.0.3         |
| parso                        | 0.8.3         |
| pathlib                      | 1.0.1         |
| pillow                       | 10.2.0        |
| pip                          | 24.2          |
| prompt-toolkit               | 3.0.43        |
| protobuf                     | 5.28.3        |
| pure-eval                    | 0.2.2         |
| py                           | 1.8.1         |
| PyQt5-sip                    | 12.8.1        |
| pytest                       | 5.4.0         |
| python-apt                   | 2.0.0+ubuntu0.20.4.8 |
| python-dateutil              | 2.8.2         |
| python-debian                | 0.1.36ubuntu1 |
| python-dotenv                | 1.0.1         |
| python-json-logger           | 2.0.7         |
| python-qt-binding            | 0.4.4         |
| python-snappy                | 0.5.3         |
| python-xlib                  | 0.33          |
| pytorch-lightning            | 1.5.10        |
| pytorch-model-summary        | 0.1.2         |
| PyYAML                       | 6.0.1         |
| pyyaml_env_tag               | 0.1           |
| pyzmq                        | 24.0.1        |
| regex                        | 2023.12.25    |
| requests                     | 2.32.3        |
| scikit-image                 | 0.19.2        |
| scikit-learn                 | 1.0.2         |
| scipy                        | 1.4.1         |
| seaborn                      | 0.12.2        |
| service-identity             | 18.1.0        |
| setuptools                   | 60.2.0        |
| simplejson                   | 3.16.0        |
| split-folders                | 0.5.1         |
| ssh-import-id                | 5.10          |
| tensorboard                  | 2.2.2         |
| tensorboard-data-server      | 0.6.1         |
| tensorboard-plugin-wit       | 1.8.0         |
| tensorboardX                 | 2.6.2.2       |
| tensorflow                   | 2.4.0         |
| tensorflow-estimator         | 2.2.0         |
| tensorflow-gpu               | 2.2.0         |
| tensorflow-io-gcs-filesystem | 0.34.0        |
| tf                           | 1.13.2        |
| tf2-geometry-msgs            | 0.7.7         |
| tf2-py                       | 0.7.7         |
| tf2-ros                      | 0.7.7         |
| threadpoolctl                | 3.3.0         |
| timm                         | 1.0.8         |
| topic-tools                  | 1.16.0        |
| torch                        | 1.10.1+cu111  |
| torchaudio                   | 0.10.1+cu111  |
| torchmetrics                 | 0.8.2         |
| torchvision                  | 0.11.2+cu111  |
| tqdm                         | 4.66.5        |
| transformers                 | 4.13.0        |
| transforms3d                 | 0.4.1         |
| typing_extensions            | 4.12.2        |
| wheel                        | 0.43.0        |
| ytsphinx                     | 1.2.1.dev20200430 |
| zipp                         | 3.1.0         |



## Acknowledgments
 We would like to express our sincere gratitude to Dr. Anil Singh Parihar (Professor, Delhi Technological University, New Delhi, INDIA) for invaluable insights, suggestions and constructive feedback. Furthermore, we would like to thank Dr. Chun-Rong Huang (Professor, National Yang Ming Chiao Tung University,  Hsinchu, Taiwan) for his support in providing the relevant files.
