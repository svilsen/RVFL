N_hidden <- 10
B <- 50
mm <- bag_rwnn(y ~ ., data = example_data, 
               N_hidden = N_hidden, lambda = 0.2, B = B)

w <- runif(B)
w <- w / sum(w)
set_weights(mm, weights = w)
