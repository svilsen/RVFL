## Models with a single hidden layer
n_hidden <- 50
lambda <- 1

# Regression
m1 <- rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda)

# Classification
m2 <- rwnn(I(y > median(y)) ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda)

## Model with multiple hidden layers
n_hidden <- c(20, 15, 10, 5)
lambda <- 0.01

# Combining outputs from all hidden layers (default)
m3 <- rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda)

# Using only the output of the last hidden layer
m4 <- rwnn(y ~ ., data = example_data, n_hidden = n_hidden,
           lambda = lambda, control = list(combine_hidden = FALSE))

