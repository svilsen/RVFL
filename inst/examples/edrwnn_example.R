N_hidden <- rep(10, 15)
ed_rwnn(y ~ ., data = example_data, 
        N_hidden = N_hidden, lambda = 0.2)
