######################################################################################
####################### AE pre-trained RVFL neural network #######################
######################################################################################

#' @title Random vector functional link
#' 
#' @description Set-up and estimate weights of a random vector functional link neural network using an auto-encoder for unsupervised pre-training of the hidden weights.
#' 
#' @param X A matrix of observed features used to train the parameters of the output layer.
#' @param y A vector of observed targets used to train the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layer.
#' @param method The penalisation type used in the auto-encoder (either \code{"l1"} or \code{"l2"}).
#' @param control A list of additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An \link{RVFL-object}.
#' 
#' @export
aeRVFL <- function(X, y, N_hidden = c(), lambda = NULL, method = "l1", control = list()) {
    UseMethod("aeRVFL")
}

#' @rdname aeRVFL
#' @method aeRVFL default
#' 
#' @example inst/examples/aervfl_example.R
#' 
#' @export
aeRVFL.default <- function(X, y, N_hidden = c(), lambda = NULL, method = "l1", control = list()) {
    ## Creating control object 
    if (length(N_hidden) > 1) {
        N_hidden <- N_hidden[1]
        warning("More than one hidden was found, but this method is designed with a single hidden layer in mind, therefore, only the first element 'N_hidden' is used.")
    }
    
    control$N_hidden <- N_hidden
    control <- do.call(control_RVFL, control)
    
    #
    bias_hidden <- control$bias_hidden
    activation <- control$activation
    N_features <- control$N_features
    rng_function <- control$rng
    rng_pars <- control$rng_pars
    
    ## Checks
    dc <- data_checks(y, X)
    
    # Regularisation
    if (is.null(lambda)) {
        lambda <- 0
        warning("Note: 'lambda' was not supplied and set to 0.")
    } else if (lambda < 0) {
        lambda <- 0
        warning("'lambda' has to be a real number larger than or equal to 0.")
    }
    
    if (length(lambda) > 1) {
        lambda <- lambda[1]
        warning("The length of 'lambda' was larger than 1, only the first element will be used.")
    }
    
    if (is.null(N_features)) {
        N_features <- ncol(X)
    }
    
    if (length(N_features) > 1) {
        N_features <- N_features[1]
        warning("The length of 'N_features' was larger than 1, only the first element will be used.")
    }
    
    if ((N_features < 1) || (N_features > dim(X)[2])) {
        stop("'N_features' have to be between 1 and the total number of features.")
    }
    
    ## Creating random weights
    nr_connections <- N_hidden * (dim(X)[2] + as.numeric(bias_hidden))
    rng_pars$n <- nr_connections
    random_weights <- do.call(rng_function, rng_pars)
    W_hidden <- list(matrix(random_weights, ncol = N_hidden))
    
    ## Values of last hidden layer (before pre-training)
    H_tilde <- rvfl_forward(X = X, W = W_hidden, activation = activation, bias = bias_hidden)
    H_tilde <- lapply(seq_along(H_tilde), function(i) matrix(H_tilde[[i]], ncol = N_hidden[i]))
    H_tilde <- do.call("cbind", H_tilde)
    
    X_tilde <- X
    if (bias_hidden) {
        X_tilde <- cbind(1, X_tilde)
    }
    
    ## Auto-encoder pre-training
    if (method == "l1") {
        W_tilde <- lasso_ls(H_tilde, X_tilde, tau = 1, max_iterations = 1000, step_shrink = 0.001)$W
        W_hidden[[1]] <- t(W_tilde)
    } else if (method == "l2") {
        HT_tilde <- t(H_tilde)
        I_tilde <- diag(ncol(H_tilde))
        
        W_tilde <- solve(HT_tilde %*% H_tilde + I_tilde) %*% HT_tilde %*% X_tilde
        W_hidden[[1]] <- t(W_tilde)
    } else {
        stop("Method not implemented, please set method to either \"l1\" or \"l2\".")
    }
    
    ## Values of last hidden layer (after pre-training)
    H <- rvfl_forward(X, W_hidden, activation, bias_hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = N_hidden[i]))
    H <- do.call("cbind", H)
    
    ## Estimate parameters in output layer
    if (control$bias_output) {
        H <- cbind(1, H)
    }
    
    O <- H
    if (control$combine_input) {
        O <- cbind(X, H)
    }
    
    W_output <- estimate_output_weights(O, y, control$lnorm, lambda)
    
    ## Return object
    object <- list(
        data = if(control$include_data) list(X = X, y = y) else NULL, 
        N_hidden = N_hidden, 
        activation = activation, 
        lambda = lambda,
        Bias = list(Hidden = bias_hidden, Output = control$bias_output),
        Weights = list(Hidden = W_hidden, Output = W_output$beta),
        Sigma = list(Hidden = NA, Output = W_output$sigma),
        Combined = control$combine_input
    )
    
    class(object) <- "RVFL"
    return(object)
}