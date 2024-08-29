n_hidden <- c(20, 15, 10, 5)
lambda <- 0.01

#
m <- ed_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda)
