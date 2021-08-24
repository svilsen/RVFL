N <- 200
p <- 5

X <- matrix(rnorm(N * p), ncol = p) 
beta <- matrix(runif(p), ncol = 1) 
y <- exp(X %*% beta) + rnorm(p)

N_hidden <- 100
sampleRVFL(X = X, y = y, N_hidden = N_hidden)
