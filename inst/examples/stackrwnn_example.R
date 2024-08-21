n_hidden <- c(20, 15, 10, 5)
lambda <- 0.01
B <- 100

## Using the average of the stack to predict new targets
\dontrun{
m <- stack_rwnn(y ~ ., data = example_data, n_hidden = n_hidden,
                lambda = lambda, B = B)
}

## Using the optimised weighting of the stack to predict new targets
\dontrun{
m <- stack_rwnn(y ~ ., data = example_data, n_hidden = n_hidden,
                lambda = lambda, B = B, optimise = TRUE)
}

