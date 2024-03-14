## RWNN
\dontrun{
m <- rwnn(y ~ ., data = example_data, N_hidden = c(10, 15), lambda = 2, control = list(lnorm = "l2"))

m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    (\(x) x$Weights)()

m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    reduce_network(method = "correlationtest", rho = 0.995, alpha = 0.05) |> 
    (\(x) x$Weights)()


m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    reduce_network(method = "correlationtest", rho = 0.995, alpha = 0.05) |> 
    reduce_network(method = "lamp", p = 0.2) |> 
    (\(x) x$Weights)()

m |> 
    reduce_network(method = "relief", p = 0.4, type = "neuron") |> 
    reduce_network(method = "relief", p = 0.4, type = "weight") |> 
    reduce_network(method = "output") |> 
    (\(x) x$Weights)()
}

## ERWNN (reduction is performed element-wise on each RWNN)
\dontrun{
m <- bag_rwnn(y ~ ., data = example_data, N_hidden = c(10, 15), lambda = 2, B = 100, control = list(lnorm = "l2"))

m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    reduce_network(method = "relief", p = 0.2, type = "weight") |> 
    reduce_network(method = "output")
}