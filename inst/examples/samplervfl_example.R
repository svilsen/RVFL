N <- 2000
p <- 5

X <- matrix(rnorm(N * p), ncol = p)
X[, 1] <- sin(X[, 1])
X[, 2] <- exp(X[, 2])
X[, 3] <- cos(X[, 3])
X[, 4] <- X[, 4] * X[, 4]

beta <- matrix(rnorm(p), ncol = 1) 
y <- X %*% beta + rnorm(N, 0, 0.1)

N_hidden <- 1000

\dontrun{
sampleRVFL(X = X, y = y, N_hidden = N_hidden, control_sample = list(method = "resample"))
sampleRVFL(X = X, y = y, N_hidden = N_hidden, control_sample = list(method = "map"))
}
