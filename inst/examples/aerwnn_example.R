N_hidden <- c(10, 10)

## Using L1-norm in the auto-encoder (sparse solution)
\dontrun{
ae_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, lambda = 0.2, method = "l1")
}

## Using L2-norm in the auto-encoder (non-sparse solution)
\dontrun{
ae_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, lambda = 0.2, method = "l2")
}