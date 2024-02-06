## Models with a single hidden layer
N_hidden <- 50
lambda <- 1

# Regression
rwnn(y ~ ., data = example_data, N_hidden = N_hidden, lambda = lambda)

# Classification
rwnn(I(y > 15) ~ ., data = example_data, N_hidden = N_hidden, lambda = lambda)

## Model with multiple hidden layers
N_hidden <- c(20, 15, 10, 5)
lambda <- 0.01

# Combining outputs from all hidden layers (default)
rwnn(y ~ ., data = example_data, N_hidden = N_hidden, lambda = lambda)

# Using only the output of the last hidden layer
rwnn(y ~ ., data = example_data, N_hidden = N_hidden, lambda = lambda, control = list(combine_hidden = FALSE))

