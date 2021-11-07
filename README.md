# RWNN
The `RWNN`-package implements a variaty of Random Weight Neural Network (RWNN). In RWNNs the weights are randomly initialised, but only the weights between last hidden-layer and the output-layer are fitted using the training data. If the activation function of the last hidden-layer is forced to be linear, and the loss-function is the sum of squared errors, then the resulting optimisation problem is equivalent to that of a linear model, making optimisation fast and efficient. Besides the standard RWNN this package also implements popular variants like extreme learning machines (ELM), sparse RWNN (spRWNN), and deep RWNN (dRWNN). It further allows for the creation of ensemble RWNNs like bagging RWNN (bagRWNN), boosting RWNN (boostRWNN), stacking RWNN (stackRWNN), and ensemble deep RWNN (edRWNN). Lastly, it allows for both L1 and L2 penalisation when estimating the output weights.

## Installation

The `RWNN`-package depends on `R` (>= 4.1), `Rcpp` (>= 1.0.4.6), `RcppArmadillo`, and `quadprog`. As the package is not available on CRAN, devtools is needed to install the package from github. 

From R, run the following commands:  
```r
install.packages("Rcpp")
install.packages("RcppArmadillo")
install.packages("quadprog")

install.packages("devtools")
devtools::install_github("svilsen/RWNN")
```

## Usage
In the following the data is randomly generated and split into training and validation sets. After which, three models are fitted: (1) a simple RWNN, (2) a bagged RWNN with equal weighting, and (3) a bagged RWNN where the ensemble weights are optimised using the validation set.

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

