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
B <- 100
bRVFL <- bagRVFL(X = X, y = y, 
                 N_hidden = N_hidden, B = B, lambda = 0.2)

w <- runif(B)
w <- w / sum(w)
set_weights(bRVFL, weights = w)
