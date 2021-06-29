N <- 200
p <- 5

X <- matrix(rnorm(N * p), ncol = p) 
beta <- matrix(runif(p), ncol = 1) 
y <- X %*% beta + rnorm(p)

N_hidden <- 10
B <- 10
epsilon <- 0.2
boostRVFL(X = X, y = y, N_hidden = N_hidden, B = B, lambda = 0.2, 
          epsilon = epsilon, combine_input = FALSE)
