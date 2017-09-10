### mcnMovingStats

A small module designed to provide code for computing network activation statistics.  This can be useful for tasks such as online estimation of the covariance of features across a dataset.

### Demo

The [run\_cov\_estimator.m](imagenet/run_cov_estimator.m) provides an example of how to estimate the means and covariances of a subset of feature activations for `ResNet-50` on a subsample of the ImageNet training data.

### Installation 

The easiest way to use this module is to install it with the `vl_contrib` 
package manager. `mcnMovingStats` can be installed with 
the following commands from the root directory of your MatConvNet 
installation:

```
vl_contrib('install', 'mcnMovingStats') ;
vl_contrib('setup', 'mcnMovingStats') ;
```

The module also requires *autonn*, which can similarly be installed with `vl_contrib` (instructions [here](https://github.com/vlfeat/autonn)).


### Notes

To provide numerical stability, the `vl_nnmovingstats()` function tracks the *shifted* moments of features, rather than their mean and covariance (which can be computed directly from the shifted moments).  Good explanations of incremental covariance estimation can be found [here](http://rebcabin.github.io/blog/2013/01/22/covariance-matrices/) and [here](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance).  Note that typically, the higher the feature layer, the longer the covariance estimation will take to converge.

Code by Samuel Albanie and David Novotny.
