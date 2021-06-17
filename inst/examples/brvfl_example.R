N <- 200
p <- 5

X <- matrix(rnorm(N * p), ncol = p) 
beta <- matrix(runif(p), ncol = 1) 
y <- X %*% beta + rnorm(p)

N_hidden <- c(10, 2, 4, 2)
B <- 100
BRVFL(X = X, y = y, N_hidden = N_hidden, B = B, combine_input = FALSE)
