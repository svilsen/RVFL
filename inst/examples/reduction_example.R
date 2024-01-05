m <- rwnn(y ~ ., data = example_data, N_hidden = 10, lambda = 200, control = list(lnorm = "l1"))

m |> 
    reduce_network(method = "lamp", p = 0.1) |> 
    (\(x) x$Weights)()

m |> 
    reduce_network(method = "lamp", p = 0.1) |>
    reduce_network(method = "last") |> 
    (\(x) x$Weights)()
