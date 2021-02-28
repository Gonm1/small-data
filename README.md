# small-data-code

#### Env instalation

To create the virtual environment run:
```
virtualenv venv
```
then activate it:
```
source venv/bin/activate
```
and run:
```
pip3 install -r requirements.txt
```
####  Data preparation
run preprocess.py to prepare the data for each program:
```
cd MNIST
python3 preprocess.py
```
Then you can run the programs on MNIST.
Example for running SVM on MNIST:
```
cd MNIST
python3 SVM.py
```

#### Datasets source
1.- [MNIST](http://yann.lecun.com/exdb/mnist/)

2.- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

3.- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
