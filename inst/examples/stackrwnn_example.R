## Using the average of the stack to predict new targets
\dontrun{
stack_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, 
           lambda = lambda, B = B)
}

## Using the optimised weighting of the stack to predict new targets
\dontrun{
stack_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, 
           lambda = lambda, B = B, optimise = TRUE)
}

