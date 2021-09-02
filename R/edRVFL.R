#######################################################################################
####################### Ensemble Deep ERVFL neural network ############################
#######################################################################################

#' @title Ensemble deep random vector functional links
#' 
#' @description Use multiple layers to create deep ensemble random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param control A list of additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An \link{ERVFL-object}.
#' 
#' @export
edRVFL <- function(X, y, N_hidden, lambda = 0, control = list()) {
    UseMethod("edRVFL")
}

#' @rdname edRVFL
#' @method edRVFL default
#' 
#' @example inst/examples/edrvfl_example.R
#' 
#' @export
edRVFL.default <- function(X, y, N_hidden, lambda = 0, control = list()) {
    ## Checks
    dc <- data_checks(y, X)
    
    ##
    deepRVFL <- RVFL(X = X, y = y, N_hidden = N_hidden, lambda = lambda, control = control)
    H <- rvfl_forward(X = X, W = deepRVFL$Weights$Hidden, activation = deepRVFL$activation, bias = deepRVFL$Bias$Hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = deepRVFL$N_hidden[i]))
    O <- lapply(seq_along(H), function(i) cbind(1, X, H[[i]]))
    beta <- lapply(seq_along(O), function(i) estimate_output_weights(O[[i]], y, lambda)$beta)
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = deepRVFL, 
        OutputWeights = beta,
        weights = rep(1L / length(N_hidden), length(N_hidden)), 
        method = "ed"
    ) 
    
    class(object) <- "ERVFL"
    return(object)
}
