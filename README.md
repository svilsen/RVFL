---
title: 'RWNN: Fast and simple non-linear modelling using random weight neural networks in R'
date: "01 January 2022"
output: html_document
bibliography: inst/paper.bib
author: Søren B. Vilsen
---

# Introduction
In recent years neural networks, and variants thereof, have seen a massive increase in popularity. This increase is largely due to the flexibility of the neural network architecture, and their accuracy when applied to highly non-linear problems. However, due to the non-linear nature of the neural network architecture training the weights of the network using gradient based optimisation procedures, like back-propagation, does not guarantee a globally optimal solution. Furthermore, optimising the weights by back-propagation can be very slow process. Therefore, various simplifications of the general feed forward neural network architecture have been proposed, including the so called random vector functional link networks, also called random weight neural networks [@Schmidt1992, @Pao1994, @Cao2006] (sometimes referred to as extreme learning machines). RWNNs randomly assigns the weights of the network between the input-layer and the last hidden-layer, focusing on training the weights between the last hidden-layer and the output-layer. This simplification makes training an RWNN as fast, simple, and efficient as training a (regularised) linear model. However, by randomly drawing the weights between the input and last hidden-layer additional instability is introduced into the network. Therefore, extensions like deep RWNN [@Henriquez2018], sparse RWNN [@Zhang2019], and ensemble deep RWNN [@Shi2021] have been proposed to combat this instability.

## Variants implemented in the package
`RWNN` is a general purpose implementation of RWNNs in `R` using `Rcpp` and `RcppArmadillo` ([@RWNN2021]). The `RWNN` package allows the user to create an RWNN of any depth, letting the user set the number of neurons and activation functions in each layer, choose the sampling distribution of the random weights, and during estimation of the output-weights it allows for Moore-Penrose inversion, $\ell_1$ regularisation, and $\ell_2$ regularisation. The network, as well as output weight estimation, is implemented in C++ through `Rcpp` and `RcppArmadillo`, making training the model simple, fast, and efficient.

Along with a standard RWNN implementation, a number of popular variations have also been implemented in the `RWNN` package: 

 - **ELM** (extreme learning machine) [@Huang2006]: A simplified version of an RWNN without a link between the input and output layer. 
 - **deep RWNN** [@Henriquez2018]: An RWNN with multiple hidden layers, where the output of each hidden-layer is included as features in the model. 
 - **sparse RWNN** [@Zhang2019]: Applies sparse auto-encoder ($\ell_1$ regularised) pre-training to reduce the number non-zero weights between the input and the hidden layer (the implementation generalises this concept to allow for both $\ell_1$ and $\ell_2$ regularisation). 
 - **ensemble deep RWNN** [@Shi2021]: An extension of deep RWNNs using the output of each hidden layer to create separate RWNNs. These RWNNs are then used to create an ensemble prediction of the target.  

Furthermore, the `RWNN` package also includes general implementations of the following ensemble methods (using RWNNs as base learners): 

 - **Stacking**: Stack multiple randomly generated RWNN's, and estimate their contribution to the weighted ensemble prediction using $k$-fold cross-validation.
 - **Bagging** [@Xin2021]: Bootstrap aggregation of RWNN's creates a number of bootstrap samples, sampled with replacement from the training-set. Furthermore, as in random forest, instead of using all features when training each RWNN, a subset of the features can be  chosen at random. 
 - **Boosting**: Gradient boosting creates a series of RWNN's, where an element, $k$, of the series is trained on the residual of the previous $k - 1$ RWNN's. It further allows for manipulation of the learning rate used to improve the generalisation of the boosted model. Lastly, like the implemented bagging method, the number of features used in each iteration can be chosen at random (also called stochastic gradient boosting). 

A Bayesian sampling approach is also implemented using a simple Metropolis-Hastings sampler to sample hidden weights from the posterior distribution of the RWNN. The sampling approach can create multiple types of output such as a single RWNN using the MAP, an ensemble RWNN using stacking, and the entire posterior of the hidden weights.  

Lastly, the `RWNN` package also includes a simple method for grid based hyperparameter optimisation using $k$-fold cross-validation.

# Installation

The `RWNN`-package depends on `R` (>= 4.1), `Rcpp` (>= 1.0.4.6), `RcppArmadillo`, and `quadprog`. As the package is not available on CRAN, devtools is needed to install the package from github. 

From R, run the following commands:  
```r
install.packages("Rcpp")
install.packages("RcppArmadillo")
install.packages("quadprog")

install.packages("devtools")
devtools::install_github("svilsen/RWNN")
```

# Usage
In the following the data is randomly generated and split into training and validation sets. After which, three models are fitted: (1) #a simple RWNN, (2) a bagged RWNN with equal weighting, and (3) a bagged RWNN where the ensemble weights are optimised using the validation set.

```r
## Data set-up
N <- 2000
p <- 5

s <- seq(0, pi, length.out = N)
X <- matrix(NA, ncol = p, nrow = N)
X[, 1] <- sin(s)
X[, 2] <- cos(s)
X[, 3] <- s
X[, 4] <- s^2
X[, 5] <- s^3

beta <- matrix(rnorm(p), ncol = 1) 
y <- X %*% beta + rnorm(N, 0, 1)

## Split data into training and validtion sets
proportion_training <- 0.7
proportion_validation <- 1L - proportion_training

index_training <- sample(N, proportion_training * N)
index_validation <- seq(N)[-index_training]

X_train <- X[index_training, ]
X_val <- X[index_validation, ]

y_train <- matrix(y[index_training, ], ncol = 1)
y_val <- matrix(y[index_validation, ], ncol = 1)

## Fitting models
N_hidden <- c(10, 10)
lambda <- 0.025

# RWNN
m_rwnn <- rwnn(X = X_train, y = y_train, N_hidden = N_hidden, lambda = lambda, 
               control = list(combine_input = TRUE))

# Bagged RWNN
B <- 100 
m_bagrwnn <- bag_rwnn(X = X_train, y = y_train, lambda = lambda, N_hidden = N_hidden, B = B,
                      control = list(combine_input = TRUE, include_data = FALSE))
             
# Bagged RWNN with trained weights        
m_bagrwnn_ew <- estimate_weights(m_bagrwnn, X_val = X_val, y_val = y_val)
```

## License

This project is licensed under the MIT License.

# References
