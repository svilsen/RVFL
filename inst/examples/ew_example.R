### 
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

##
train_fraction <- 0.7
train_index <- sample(N, train_fraction * N)

X_train <- X[train_index, ]
X_val <- X[-train_index, ]

y_train <- matrix(y[train_index, ], ncol = 1)
y_val <- matrix(y[-train_index, ], ncol = 1)

###
N_hidden <- 10
B <- 100
mm <- bagRVFL(X = X_train, y = y_train, 
              N_hidden = N_hidden, lambda = 0.2, B = B)

###
estimate_weights(mm, X_val = X_val, y_val = y_val)
