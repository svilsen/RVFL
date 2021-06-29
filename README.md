# RVFL
The `RVFL`-package implements a variaty of Random Vector Functional Link (RVFL) Neural Network. In RVFL neural networks the weights are randomly initialised, but only the weights between last hidden-layer and the output-layer are fitted using the training data. If the activation function of the last hidden-layer is forced to be linear, and the loss-function is the sum of squared errors, then the resulting optimisation problem is equivalent to that of a linear model, making optimisation fast and efficient. 

## Installation

The `RVFL`-package depends on `R` (>= 4.1), `Rcpp` (>= 1.0.4.6), `RcppArmadillo`, and `Rsolnp`. As the package is not available on CRAN, devtools is needed to install the package from github. 

From R, run the following commands:  

```r
install.packages("Rcpp")
install.packages("RcppArmadillo")
install.packages("Rsolnp")

install.packages("devtools")
devtools::install_github("svilsen/RVFL")
```

## Usage
In the following the data is randomly generated and split into training and validation sets. After which, three models are fitted: (1) a simple RVFL, (2) a bagged RVFL with equal weighting, and (3) a bagged RVFL where the ensemble weights are optimised using the validation set.

```r
## Data set-up
# Number of observations
N <- 500

# Number of features
p <- 50 

# Features 
X <- matrix(rnorm(p * N), ncol = p) 

# Response
y <- matrix(
    0.2 * sin(X[, 1]) + 0.8 * exp(X[, 2]) + 2 * cos(X[, 3]) +
        1.2 * X[, 4] + 0.6 * abs(X[, 5]) + X[, 6:p] %*% rnorm(p - 5) + 
        rnorm(N, sd = 0.5), 
    ncol = 1
)

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
# Number of hidden-layers and neurons in each layer.
N_hidden <- 10
m1 <- RVFL(X = X_train, y = y_train, N_hidden = N_hidden, lambda = 0.3, combine_input = TRUE)

# Number of bootstrap samples
B <- 100 
m2 <- BRVFL(X = X_train, y = y_train, N_hidden = N_hidden, B = B, lambda = 0.3, combine_input = TRUE, include_data = FALSE)
m3 <- estimate_weights(m2, X_val = X_val, y_val = y_val)

```

## License

This project is licensed under the MIT License.

