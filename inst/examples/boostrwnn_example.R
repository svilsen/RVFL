n_hidden <- 10
B <- 100
epsilon <- 0.01
lambda <- 0.1

\dontrun{
m <- boost_rwnn(y ~ ., data = example_data, n_hidden = n_hidden,
                lambda = lambda, B = B, epsilon = epsilon)
}