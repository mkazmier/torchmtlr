# Multi-task logistic regression for individual survival prediction
PyTorch implementation of multi-task logistic regression for survival prediction.

## Setup

### Google Colaboratory
Simply follow [the Colab link](https://colab.research.google.com/drive/1o3v_9NBUYD09a2LS5ElTqXFJ0IW4CjMY?usp=sharing) and log in with your Google account. To speed up training, make sure to use the GPU runtime.

### On local machine
1. Clone or download the repo.
2. Install the required packages:
```
pip install -r requirements.txt
```
Note: by default, the CPU version of Pytorch is installed. If you want to use a local GPU, you need to [install CUDA and Pytorch with GPU support](https://pytorch.org/get-started/locally/).
3. Launch the notebook server and open the notebook.

## References
1. C.-N. Yu, R. Greiner, H.-C. Lin, and V. Baracos, ‘Learning patient-specific cancer survival distributions as a sequence of dependent regressors’, in Advances in neural information processing systems 24, pp. 1845–1853.
2. P. Jin, ‘Using Survival Prediction Techniques to Learn Consumer-Specific Reservation Price Distributions’, University of Alberta, Edmonton, AB, 2015.
3. S. Fotso, ‘Deep Neural Networks for Survival Analysis Based on a Multi-Task Framework’, arXiv:1801.05512 [cs, stat], Jan. 2018, Accessed: Feb. 11, 2020. [Online]. Available: http://arxiv.org/abs/1801.05512.
