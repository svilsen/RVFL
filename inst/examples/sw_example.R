\dontrun{
N_hidden <- 10
B <- 50

m <- bag_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, 
               lambda = 0.2, B = B)

w <- runif(B)
w <- w / sum(w)
m <- m |> set_weights(weights = w)
}