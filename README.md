# Asymptotic AURC
This is the code for " A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC) and Rates of Finite Sample Estimators".
#### Prepare dataset
* ImageNet (ILSVRC2012)
This dataset can be found on the official website if you are affiliated with a research organization. It is also available on Academic torrents.
Download the ILSVRC2012 train and validation dataset and extract those images under the folder './data/ILSVRC2012'.

* CIFAR10/100 and Amazon (https://github.com/datapythonista/mnist)
We use the pre-trained model and their test results located under the folder './results' when comparing different estimators for AURC.

#### Visualize the performance of those AURC estimators
The following code can be used to evaluate their performance on the Amazon dataset.
```bash
cd evaluate
python amazon.py
```
Then you can obtain the outputs under the folder './outputs' that contains figures like below:
<img src="https://github.com/han678/AsymptoticAURC/blob/c78db47a506fc9db5fbdcddd08f4b593c48c6a60/outputs/bias/amazon_bert.png" alt="figure" width="400">
