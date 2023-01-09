## Models with a single hidden layer
N_hidden <- 100
elm(y ~ ., data = example_data, N_hidden = N_hidden)
rwnn(y ~ ., data = example_data, N_hidden = N_hidden)

## Model with multiple hidden layers
N_hidden <- c(10, 20, 10, 5)
rwnn(y ~ ., data = example_data, N_hidden = N_hidden)
