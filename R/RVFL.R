############################################################################
####################### A simple RVFL neural network #######################
############################################################################

#' @title RVFL control function
#' 
#' @description A function used to create a control-object for the \link{RVFL} function.
#' 
#' @param hidden_bias A vector of TRUE/FALSE values. The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param activation A vector of activation functions.
#' @param weight_method The method used for generating random weights.
#' @param weight_mean The mean of the randomly initialised weights.
#' @param weight_sd The standard deviation of the randomly initialised weights.
#' @param trace An integer passed to the optimiser. If set larger than 0, a trace of the optimisation iterations is shown every '\code{trace}' iterations. 
#' 
#' @return A list of control variables.
#' @export
control_RVFL <- function(hidden_bias = TRUE, activation = NULL, 
                         weight_method = "normal", weight_mean = 0, weight_sd = 1, 
                         output_bias = TRUE, combine_input = FALSE,
                         trace = 0) {
    return(list(hidden_bias = hidden_bias, activation = activation, 
                weight_method = weight_method, weight_mean = weight_mean, weight_sd = weight_sd,
                output_bias = output_bias, combine_input = combine_input,
                trace = trace))
}


#' @title Random vector functional link
#' 
#' @description Set-up and estimate weights of a random vector functional link neural network.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
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
#' 
#' @examples X <- matrix(rnorm(100), ncol = 5) 
#' beta <- matrix(c(0.4, 1, 3, 0.2, 1.6), ncol = 1) 
#' y <- X %*% beta + rnorm(20)
#' N_hidden <- c(10, 2, 4)
#' 
#' @export
RVFL.default <- function(X, y, N_hidden, ...) {
    ## Creating control object 
    dots <- list(...)
    control <- do.call(control_RVFL, dots)
    
    rweights <- switch(control$weight_method, 
                       "normal" = rnorm)
    
    ## Checks
    if (length(control$hidden_bias) == 1) {
        hidden_bias <- rep(control$hidden_bias, length(N_hidden))
    }
    else if (length(control$hidden_bias) == length(N_hidden)) {
        hidden_bias <- control$hidden_bias
    }
    else {
        stop("The 'hidden_bias' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")
    }
    
    ## Initialisation
    X_dim <- dim(X)
    
    W_hidden <- vector("list", length = length(N_hidden))
    for (w in seq_along(W_hidden)) {
        if (w == 1) {
            nr_connections <- N_hidden[w] * (X_dim[2] + as.numeric(hidden_bias[w]))
        }
        else {
            nr_connections <- N_hidden[w] * (N_hidden[w - 1] + as.numeric(hidden_bias[w]))
        }
        
        random_weights <- rweights(nr_connections, mean = control$weight_mean, sd = control$weight_sd)
        W_hidden[[w]] <- matrix(random_weights, ncol = N_hidden[w])
    }
    
    ## Values of last hidden layer
    H <- RFRVFL:::rvfl_forward(X, W_hidden, hidden_bias)
    
    ## Estimate parameters in output layer
    if (control$output_bias) {
        H <- cbind(1, H)
    }
    
    O <- H
    if (control$combine_input) {
        O <- cbind(H, X)
    }
    
    W_output <- RFRVFL:::estimate_output_weights(O, y)
    
    ## Return object
    object <- list(
        data = list(X = X, y = y), 
        N_hidden = N_hidden, 
        Weights = list(Hidden = W_hidden, Output = W_output$beta),
        SE = list(Hidden = NA, Output = W_output$se)
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

