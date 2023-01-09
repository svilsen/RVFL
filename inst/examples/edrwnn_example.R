N_hidden <- rep(10, 15)
\dontrun{
ed_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, lambda = 0.2)
}