N_hidden <- 100
B <- 1000
lambda <- 0.2
\dontrun{
bag_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, lambda = lambda, B = B)
}
