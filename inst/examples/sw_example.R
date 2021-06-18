N <- 200
p <- 5

X <- matrix(rnorm(N * p), ncol = p) 
beta <- matrix(runif(p), ncol = 1) 
y <- X %*% beta + rnorm(p)

N_hidden <- 10
B <- 100
mm <- BRVFL(X = X, y = y, N_hidden = N_hidden, B = B, combine_input = FALSE)

w <- runif(B)
w <- w / sum(w)
set_weights(mm, weights = w)