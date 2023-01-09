hyperparameters <- list(
    N_hidden = list(c(100, 100, 50, 50), c(100, 10, 10), c(100)), 
    lambda = exp(seq(-8, 2))
)

folds <- 20

\dontrun{
tune_hyperparameters(
    y ~ ., data = example_data, 
    rwnn, 
    folds = folds, hyperparameters = hyperparameters, 
    trace = 1
)
}
