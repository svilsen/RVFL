N <- 200
p <- 5

X <- matrix(rnorm(N * p), ncol = p) 
beta <- matrix(runif(p), ncol = 1) 
y <- X %*% beta + rnorm(N)

N_hidden <- rep(10, 15)
edRVFL(X = X, y = y, N_hidden = N_hidden, lambda = 0.2)
