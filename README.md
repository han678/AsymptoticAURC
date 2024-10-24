# Asymptotic AURC
This is the code for "A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC) and Rates of Finite Sample Estimators" [[arxiv](https://arxiv.org/pdf/2410.15361)].
#### Prepare dataset
* ImageNet (ILSVRC2012)
This dataset can be found on the official website if you are affiliated with a research organization. It is also available on Academic torrents.
Download the ILSVRC2012 train and validation dataset and extract those images under the folder './data/ILSVRC2012'.

* CIFAR10/100 and Amazon (https://github.com/datapythonista/mnist)
We use the outputs of the pre-trained models located under the folder './results' when comparing different estimators for AURC.

#### Visualize the performance of those AURC estimators
The following code can be used to evaluate their performance on the Amazon dataset.
```bash
cd evaluate
python amazon.py
```
Then you can obtain the outputs under the folder './outputs' that contains figures like below:
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/han678/AsymptoticAURC/blob/c78db47a506fc9db5fbdcddd08f4b593c48c6a60/outputs/bias/amazon_bert.png" alt="figure" width="260">
  <img src="https://github.com/han678/AsymptoticAURC/blob/0071990151584e99ad818bd4961d27e9a49e78af/outputs/mse/amazon_bert.png" alt="figure" width="260">
  <img src="https://github.com/han678/AsymptoticAURC/blob/0071990151584e99ad818bd4961d27e9a49e78af/outputs/csf/amazon_bert.png" alt="figure" width="240">
</div>
