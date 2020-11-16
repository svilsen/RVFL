N <- 1000
p <- 5

X <- matrix(rnorm(N), ncol = p) 
beta <- matrix(runif(p), ncol = 1) 
y <- X %*% beta + rnorm(p)

N_hidden <- c(10, 2, 4, 2)
B <- 1000
BRVFL(X = X, y = y, N_hidden = N_hidden, B = B, combine_input = FALSE)