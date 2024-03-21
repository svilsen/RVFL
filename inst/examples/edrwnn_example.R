n_hidden <- rep(10, 15)
\dontrun{
m <- ed_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = 0.2)
}