n_hidden <- 50
B <- 100
lambda <- 0.01

# Regression
\dontrun{
m <- bag_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda, B = B)
}

# Classification
\dontrun{
m <- bag_rwnn(I(y > 15) ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda, B = B)
}

