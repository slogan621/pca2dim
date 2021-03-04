# pca2dim

This is a port of a matlab program that does PCA on 2 dimension data
(age/height data for young males). It is intended only to show how 
matlab code might be ported to Python using matplotlib and numpy packages.

# How to run the examples

If necessary, install required libraries using pip:

```
pip3 install matplotlib
pip3 install numpy
```

You can figure out if you actually need to do the above by trying to
load them in the Python repl:

```
slogan@cpp-learning--virtual-machine:~$ python3
Python 3.8.5 (default, Jan 27 2021, 15:41:15) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import matplotlib
>>> import numpy
>>> 
```

Next, git clone this repo:

```
slogan@cpp-learning--virtual-machine:/tmp$ git clone https://github.com/slogan621/pca2dim.git
Cloning into 'pca2dim'...
slogan@cpp-learning--virtual-machine:/tmp$
```

Then run:

```
slogan@cpp-learning--virtual-machine:~/tmp$ cd pca2dim
slogan@cpp-learning--virtual-machine:~/tmp/pca2dim$ python3 pca_2dim.py 
Matrix of Covariance
[[2.92453364 0.18682262]
 [0.18682262 0.01390859]]
Major Principal Component
[-0.99796309 -0.06379393]
End of the program
```

Several figures should display.

# Bugs

Create an issue if you run into any problems. If you have improvements, I'll
gladly respond to pull requests.


