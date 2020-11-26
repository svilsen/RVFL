# BRVFL
The `BRVFL`-package is an `R` implementation of a Bootstrap aggregated (Bagged) Random Vector Functional Link (RVFL) Neural Network. In RVFL neural networks the weights are randomly initialised, but only the weights between last hidden-layer and the output-layer are fitted using the training data. If the activation function of the last hidden-layer is forced to be linear, and the loss-function is the sum of squared errors, then the resulting optimisation problem is equivalent to that of a linear model, making optimisation fast and efficient. 

## Installation

The `BRVFL`-package depends on `R` (>= 4.1), `Rcpp` (>= 1.0.4.6), `RcppArmadillo`, and `Rsolnp`. As the package is not available on CRAN, devtools is needed to install the package from github. 

From R, run the following commands:  

```r
install.packages("Rcpp")
install.packages("RcppArmadillo")
install.packages("Rsolnp")

install.packages("devtools")
devtools::install_github("svilsen/BRVFL")
```

## Usage
In the following the data is randomly generated and split into training and validation sets. After which, three models are fitted: (1) a simple RVFL, (2) a bagged RVFL with equal weighting, and (3) a bagged RVFL where the ensemble weights are optimised using the validation set.

```r
## Data set-up
N <- 500 # Number of observations
p <- 50 # Number of features
X <- matrix(rnorm(p * N), ncol = p) # Features 
y <- matrix(
    0.2 * sin(X[, 1]) + 0.8 * exp(X[, 2]) + 2 * cos(X[, 3]) +
        1.2 * X[, 4] + 0.6 * abs(X[, 5]) + X[, 6:p] %*% rnorm(p - 5) + 
        rnorm(N, sd = 0.5), 
    ncol = 1
) # Response

## Split data into training and validtion sets
proportion_training <- 0.7
proportion_validation <- 1L - proportion_training

index_training <- sample(N, proportion_training * N)
index_validation <- seq(N)[-index_training]

training_X <- X[index_training, ]
validation_X <- X[index_validation, ]

training_y <- matrix(y[index_training, ], ncol = 1)
validation_y <- matrix(y[index_validation, ], ncol = 1)

## Fitting models
N_hidden <- c(100, 25, 12, 15) # Number of hidden-layers and neurons in each layer.
m1 <- RVFL(X = training_X, y = training_y, N_hidden = N_hidden, combine_input = TRUE)

B <- 100 # Number of bootstrap samples
m2 <- BRVFL(X = training_X, y = training_y, N_hidden = N_hidden, B = B, combine_input = TRUE)
m3 <- estimate_weights(m2, validation_X = validation_X, validation_y = validation_y)
```

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is partly financed by the “CloudBMS – The New Generation of Intelligent Battery Management Systems” research and development project, project number 64017-05167. The authors gratefully acknowledge EUDP Denmark for providing the financial support necessary for carrying out this work.
