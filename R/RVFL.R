############################################################################
####################### A simple RVFL neural network #######################
############################################################################

#' @title RVFL control function
#' 
#' @description A function used to create a control-object for the \link{RVFL} function.
#' 
#' @param activation A vector of activation functions.
#' @param weight_method The method used for generating random weights.
#' @param weight_mean The mean of the randomly initialised weights.
#' @param weight_sd The standard deviation of the randomly initialised weights.
#' @param trace An integer passed to the optimiser. If set larger than 0, a trace of the optimisation iterations is shown every '\code{trace}' iterations. 
#' 
#' @return A list of control variables.
#' @export
control_RVFL <- function(activation = NULL, weight_method = "normal", weight_mean = 0, weight_sd = 1, trace = 0) {
    return(list(activation = activation, 
                weight_method = weight_method, weight_mean = weight_mean, weight_sd = weight_sd, 
                trace = trace))
}


#' @title Random vector functional link
#' 
#' @description Set-up and estimate weights of a random vector functional link neural network.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' 
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' 
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' 
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An RVFL-object containing the random and fitted weights of the RVFL-model.
#' 
#' @export
RVFL <- function(X, y, N_hidden, ...) {
    UseMethod("RVFL")
}

#' @rdname RVFL
#' @method RVFL default
#' @export
RVFL.default <- function(X, y, N_hidden, ...) {
    ## Creating control object 
    dots <- list(...)
    control <- do.call(control_RVFL, dots)
    
    rweights <- switch(control$weight_method, 
                       "normal" = rnorm)
    
    ## Checks
    
    ## Initialisation
    X_dim <- dim(X)
    
    W_hidden <- vector("list", length = length(N_hidden))
    for (w in seq_along(weights)) {
        if (w == 1) {
            nr_connections <- N_hidden[w] * X_dim[2]
        }
        else {
            nr_connections <- N_hidden[w] * N_hidden[w - 1]
        }
        
        random_weights <- rweights(nr_connections, mean = control$weight_mean, sd = control$weight_sd)
        W_hidden[[w]] <- matrix(random_weights, ncol = N_hidden[w])
    }
    
    ## Values of last hidden layer
    H <- rvfl_forward(X, W_hidden)
    
    ## Estimate parameters in output layer
    O <- cbind(X, H)
    W_output <- estimate_output_weights(O, y, trace = control$trace)
    
    ## Return object
    object <- list(
        data = list(X = X, y = y), 
        N_hidden = N_hidden, 
        Weights = list(Hidden = W_hidden, Output = W_output)
    )
    
    class(object) <- "RVFL"
    return(object)
}


#' @title Predicting targets of an RVFL object.
#' 
#' @param object An RVFL-object.
#' @param ... Additional arguments.
#' 
#' @rdname predict
#' @method predict RVFL
#' @export
predict.RVFL <- function(object, ...) {
    return(NULL)
}

