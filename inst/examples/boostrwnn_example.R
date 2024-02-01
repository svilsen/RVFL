N_hidden <- 10
B <- 100
epsilon <- 0.01
lambda <- 0.1

\dontrun{
boost_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, 
           lambda = lambda, B = B, epsilon = epsilon)
}