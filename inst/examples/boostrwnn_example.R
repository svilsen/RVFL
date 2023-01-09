N_hidden <- 5
B <- 1000
epsilon <- 0.1
lambda <- 0.1

\dontrun{
boost_rwnn(y ~ ., data = example_data, N_hidden = N_hidden, 
           lambda = lambda, B = B, epsilon = epsilon)
}