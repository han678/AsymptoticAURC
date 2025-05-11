This repository contains the code for the paper **"A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC) and Rates of Finite Sample Estimators"** [[arxiv](https://arxiv.org/pdf/2410.15361)].

#### Key Dependencies
To run the code, you will need the following dependencies (excluding common packages like `scipy`, `numpy`, and `torch`):

- **Python** ≥ 3.8
- **timm**: A library for PyTorch models pre-trained on the ImageNet dataset. [Learn more here](https://timm.fast.ai).

  Install via pip:
  ```bash
  pip install timm
  ```

#### Preparing Datasets and Models

- **ImageNet (ILSVRC2012)**:
  - The ImageNet dataset can be obtained from the official website if you are affiliated with a research organization. It is also available on Academic Torrents.
  - Download the ILSVRC2012 validation set and extract the images into the `data/ILSVRC2012` folder. This validation set is used to compare the performance of the AURC estimators.

- **CIFAR-10/100 and Amazon Datasets**:
  - For the CIFAR-10/100 and Amazon datasets, we use the outputs of pre-trained models on their test sets, which are located in the `results` folder for comparison across different AURC estimators.
  - Pre-trained models for CIFAR-10/100 can be downloaded from [Zenodo](https://zenodo.org/records/10724791). Place the downloaded files in the `results/cifar` folder.
  - The outputs of the pre-trained models for the Amazon dataset can be found in the `results/Amazon` folder.

#### Using the AURC estimator in your project

To evaluate AURC using our estimator, you can copy the file `utils/estimators.py` into your repository. If you want to use it as a loss function, you can copy the file `utils/loss.py` into your repository.

#### Visualizing the performance of AURC estimators

To evaluate the performance of AURC estimators on the Amazon dataset, use the following commands:

```bash
cd evaluate
python amazon.py
```
 
After running the script, the results will be saved in the `outputs` folder, which contains figures visualizing the estimator performance, as shown below:
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/han678/AsymptoticAURC/blob/63c7630c11a15e37f1c3bf7d454e25fb3fcf84d0/outputs/bias/amazon_bert.png" alt="Bias Figure" width="260">
  <img src="https://github.com/han678/AsymptoticAURC/blob/63c7630c11a15e37f1c3bf7d454e25fb3fcf84d0/outputs/mse/amazon_bert.png" alt="MSE Figure" width="260">
  <img src="https://github.com/han678/AsymptoticAURC/blob/63c7630c11a15e37f1c3bf7d454e25fb3fcf84d0/outputs/csf/amazon_bert.png" alt="CSF Figure" width="240">
</div>
 
These figures help visualize the bias, mean squared error (MSE), and different confidence score functions (CSFs) for the AURC estimators on the Amazon dataset.

#### Reference
If you found this work or code useful, please cite:

```
@inproceedings{zhou2024novel,
      title={A Novel Characterization of the Population Area Under the 
Risk Coverage Curve (AURC) and Rates of Finite Sample Estimators}, 
      author={Han Zhou and Jordy Van Landeghem and Teodora Popordanoska 
and Matthew B. Blaschko},
      booktitle={International Conference on Machine Learning},
      year={2025},
      eprint={2410.15361},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2410.15361}, 
}
```
#### License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
