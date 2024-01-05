N_hidden <- 50
B <- 100
lambda <- 0.01

# Regression
\dontrun{
bag_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, lambda = lambda, B = B)
}

# Classification
\dontrun{
bag_rwnn(I(y > 15) ~ ., data = example_data, N_hidden = N_hidden, lambda = lambda, B = B)
}

