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

hyperparameters <- list(
    N_hidden = list(c(100, 100, 50, 50), c(100, 10, 10), c(100)), 
    lambda = exp(seq(-12, 4))
)

folds <- 20

\dontrun{
tune_hyperparameters(rwnn, X, y, folds, hyperparameters)
}