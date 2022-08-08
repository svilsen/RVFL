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

N_hidden <- 10
lambda <- 1

## Returning an RWNN object using just the MAP estimate of the weights
\dontrun{
mh_rwnn(y ~ X, N_hidden = N_hidden, 
        lambda = lambda, control = list(method = "map"))
}

## Returning an ERWNN object resampling weights from the created posterior sample
\dontrun{
mh_rwnn(y ~ X, N_hidden = N_hidden, 
        lambda = lambda, control = list(method = "stack"))
}

## Returning an SRWNN object of the sampled posterior
\dontrun{
mh_rwnn(y ~ X, N_hidden = N_hidden, 
        lambda = lambda, control = list(method = "posterior"))
}
