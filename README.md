# STSA - Capturing Spatio-Temporal Dependencies with Competitive Set Attention for Video Summarization  
[![DOI](https://zenodo.org/badge/882984239.svg)](https://doi.org/10.5281/zenodo.14032948)
The Official Github Repository of "Capturing Spatio-Temporal Dependencies with Competitive Set Attention for Video Summarization". 

- This paper has been accepted in [The Visual Computer](https://doi.org/10.1007/s00371-025-03865-1).

### Download the datasets
- [Dataset](https://drive.google.com/drive/folders/1KTpftiMchP0q-pdcJ4K7HaARJloOwQA6?usp=sharing)
- Create a datasets folder inside the ./STSA/datasets folder and save the downloaded .h5 files of the datasets.
  
- [Requirements](https://github.com/ara-47/STSA/blob/main/requirements.txt)

### Create STSA conda environment
```
conda create -n STSA
```

### Activate STSA environment
```
conda activate STSA
```

### Install the packages provided in the requirement.txt
```
pip install -r requirements.txt
```

### To train the model
```
python3 train.py --exp_name 'ExperimentName' --dataset 'DatasetName(TVSum or SumMe)' --batch_size #BATCH_SIZE --epochs #EPOCHS
```

### Evaluation
- The evaluation is provided in the training file.

### Citation
```
@article{arafat2025capturing,
  title={Capturing spatiotemporal dependencies with competitive set attention for video summarization},
  author={Arafat, Md Hasnat Hosen and Singh, Kavinder},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer}
}
```

### Acknowledgments
```
We would like to express our sincere gratitude to Dr. Anil Singh Parihar (Professor, Delhi Technological University, New Delhi, INDIA) for invaluable insights, suggestions and constructive feedback. Furthermore, we would like to thank Dr. Chun-Rong Huang (Professor, National Yang Ming Chiao Tung University,  Hsinchu, Taiwan) for his support in providing the relevant files.
```

### Send us feedback
- If you have any queries or feedback, please contact us @(**kavinder@dtu.ac.in**).
