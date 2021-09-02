### 
N <- 200
p <- 5

X <- matrix(rnorm(N * p), ncol = p) 
beta <- matrix(runif(p), ncol = 1) 
y <- X %*% beta + rnorm(N)

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
              N_hidden = N_hidden, B = B, lambda = 0.2)

###
estimate_weights(mm, X_val = X_val, y_val = y_val)
