n_hidden <- c(10, 10)
lambda <- c(2, 0.2)

## Using L1-norm in the auto-encoder (sparse solution)
\dontrun{
m <- ae_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda, method = "l1")
}

## Using L2-norm in the auto-encoder (dense solution)
\dontrun{
m <- ae_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda, method = "l2")
}