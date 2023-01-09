##
N <- dim(example_data)[1]
train_fraction <- 0.7
train_index <- sample(N, train_fraction * N)

data_train <- example_data[train_index, ]
data_val <- example_data[-train_index, ]

###
N_hidden <- 10
B <- 100
mm <- bag_rwnn(y ~ ., data = data_train, 
               N_hidden = N_hidden, lambda = 0.2, B = B)

###
estimate_weights(mm, X_val = data_val[, -1], y_val = data_val[, 1])
