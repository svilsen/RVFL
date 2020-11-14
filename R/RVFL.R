############################################################################
####################### A simple RVFL neural network #######################
############################################################################

#' @title RVFL control function
#' 
#' @description A function used to create a control-object for the \link{RVFL} function.
#' 
#' @param bias_hidden A vector of TRUE/FALSE values. The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param activation A vector of activation functions (NOT IMPLEMENTED -- IN THIS VERSION ALL ACTIVATIONS ARE SIGMOID).
#' @param bias_output TRUE/FALSE: Should a bias be added to the output layer?
#' @param combine_input TRUE/FALSE: Should the input and hidden layer be combined for the output layer?
#' 
#' @return A list of control variables.
#' @export
control_RVFL <- function(bias_hidden = TRUE, activation = NULL, 
                         bias_output = TRUE, combine_input = FALSE) {
    return(list(bias_hidden = bias_hidden, activation = activation, 
                bias_output = bias_output, combine_input = combine_input))
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
#' @example inst/examples/rvfl_example.R
#' 
#' @export
RVFL.default <- function(X, y, N_hidden, ...) {
    ## Creating control object 
    dots <- list(...)
    control <- do.call(control_RVFL, dots)
    
    ## Checks
    if (length(N_hidden) < 1) {
        stop("When the number of hidden layers is equal to 0, this model reduces to a linear model, ?lm.")
    }
    
    if (length(control$bias_hidden) == 1) {
        bias_hidden <- rep(control$bias_hidden, length(N_hidden))
    }
    else if (length(control$bias_hidden) == length(N_hidden)) {
        bias_hidden <- control$bias_hidden
    }
    else {
        stop("The 'bias_hidden' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")
    }
    
    ## Initialisation
    X_dim <- dim(X)
    
    W_hidden <- vector("list", length = length(N_hidden))
    for (w in seq_along(W_hidden)) {
        if (w == 1) {
            nr_connections <- N_hidden[w] * (X_dim[2] + as.numeric(bias_hidden[w]))
        }
        else {
            nr_connections <- N_hidden[w] * (N_hidden[w - 1] + as.numeric(bias_hidden[w]))
        }
        
        random_weights <- runif(nr_connections, -1, 1)
        W_hidden[[w]] <- matrix(random_weights, ncol = N_hidden[w])
    }
    
    ## Values of last hidden layer
    H <- rvfl_forward(X, W_hidden, bias_hidden)
    
    ## Estimate parameters in output layer
    if (control$bias_output) {
        H <- cbind(1, H)
    }
    
    O <- H
    if (control$combine_input) {
        O <- cbind(H, X)
    }
    
    W_output <- estimate_output_weights(O, y)
    
    ## Return object
    object <- list(
        data = list(X = X, y = y), 
        N_hidden = N_hidden, 
        Bias = list(Hidden = bias_hidden, Output = control$bias_output),
        Weights = list(Hidden = W_hidden, Output = W_output$beta),
        SE = list(Hidden = NA, Output = W_output$se), 
        Sigma = list(Hidden = NA, Output = W_output$sigma),
        Combined = control$combine_input
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
    dots <- list(...)
    
    if (is.null(dots$newdata)) {
        newdata <- object$data$X
    }
    else {
        if (dim(dots$newdata)[2] != dim(object$data$X)[2]) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
        
        newdata <- dots$newdata 
    }
    
    newH <- rvfl_forward(
        X = newdata, 
        W = object$Weights$Hidden, 
        bias = object$Bias$Hidden
    )
    
    ## Estimate parameters in output layer
    if (object$Bias$Output) {
        newH <- cbind(1, newH)
    }
    
    newO <- newH
    if (object$Combined) {
        newO <- cbind(newH, newdata)
    }
    
    newy <- newO %*% object$Weights$Output
    return(newy)
}

#' @title Residuals of the RVFL object.
#' 
#' @param object An RVFL-object.
#' @param ... Additional arguments.
#' 
#' @rdname residuals
#' @method residuals RVFL
#' @export
residuals.RVFL <- function(object, ...) {
    dots <- list(...)
    newy <- predict.RVFL(object, newdata = NULL, ...)
    
    r <- newy - object$data$y
    if (tolower(dots$type) %in% c("standard", "standardised", "rs")) {        
        r <- r / object$Sigma$Output
    }
    
    return(r)
}

#' @title Coefficients of the RVFL object.
#' 
#' @param object An RVFL-object.
#' @param ... Additional arguments.
#' 
#' @rdname coef
#' @method coef RVFL
#' @export
coef.RVFL <- function(object, ...) {
    return(object$Weights$Output)
}

