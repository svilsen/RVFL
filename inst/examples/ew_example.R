N <- nrow(example_data)

train_fraction <- 0.7
train_index <- sample(N, train_fraction * N)

data_train <- example_data[train_index, ]
data_val <- example_data[-train_index, ]

###
n_hidden <- 10
B <- 100
lambda <- 0.1

mm <- boost_rwnn(y ~ ., data = data_train, n_hidden = n_hidden, lambda = lambda, B = B)

###
estimate_weights(mm, data_val = data_val)
