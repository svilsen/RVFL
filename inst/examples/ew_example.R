### 
N <- 200
p <- 5

X <- matrix(rnorm(N * p), ncol = p) 
beta <- matrix(runif(p), ncol = 1) 
y <- X %*% beta + rnorm(p)

proportion_training <- 0.7
proportion_validation <- 1L - proportion_training
index_training <- sample(N, 0.7 * N)

training_X <- X[index_training, ]
validation_X <- X[-index_training, ]

training_y <- matrix(y[index_training, ], ncol = 1)
validation_y <- matrix(y[-index_training, ], ncol = 1)

###
N_hidden <- c(10, 2, 4, 2)
B <- 100
mm <- BRVFL(X = training_X, y = training_y, N_hidden = N_hidden, 
            B = B, combine_input = FALSE)

###
estimate_weights(mm, validation_X = validation_X, validation_y = validation_y)