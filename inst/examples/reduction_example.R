## RWNN-object
n_hidden <- c(10, 15)
lambda <- 2

\dontrun{
m <- rwnn(y ~ ., data = example_data, n_hidden = n_hidden, 
          lambda = lambda, control = list(lnorm = "l2"))

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

## ERWNN-object (reduction is performed element-wise on each RWNN)
n_hidden <- c(10, 15)
lambda <- 2
B <- 100

\dontrun{
m <- bag_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, 
              lambda = lambda, B = B, control = list(lnorm = "l2"))

m |> 
    reduce_network(method = "relief", p = 0.2, type = "neuron") |> 
    reduce_network(method = "relief", p = 0.2, type = "weight") |> 
    reduce_network(method = "output")
}