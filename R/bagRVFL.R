############################################################################
####################### Bagging ERVFL neural network #######################
############################################################################

#' @title Bagging random vector functional links.
#' 
#' @description Use bootstrap aggregation to reduce the variance of random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of bootstrap samples.
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param control A list of additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An \link{ERVFL-object}.
#' 
#' @export
bagRVFL <- function(X, y, N_hidden, B = 100, lambda = 0, control = list()) {
    UseMethod("bagRVFL")
}

#' @rdname bagRVFL
#' @method bagRVFL default
#' 
#' @example inst/examples/bagrvfl_example.R
#' 
#' @export
bagRVFL.default <- function(X, y, N_hidden, B = 100, lambda = 0, control = list()) {
    ## Checks
    dc <- data_checks(y, X)
    
    if (is.null(B)) {
        B <- 100
        warning(paste0("Note: 'B' was not supplied, 'B' was set to ", B, "."))
    }
    
    if (is.null(control$N_features)) {
        control$N_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        indices_b <- sample(nrow(X), nrow(X), replace = TRUE)
        
        X_b <- matrix(X[indices_b, ], ncol = ncol(X))
        y_b <- matrix(y[indices_b], ncol = ncol(y))    
        
        # rvfl_b <- sampleRVFL(
        #     X = X_b, y = y_b, N_hidden = N_hidden, 
        #     control_rvfl = control, 
        #     control_sample = list(method = "map")
        # )
        
        rvfl_b <- RVFL(X = X_b, y = y_b, N_hidden = N_hidden, lambda = lambda, control = control)
        objects[[b]] <- rvfl_b
    }
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = objects, 
        weights = rep(1L / B, B), 
        method = "bagging"
    )  
    
    class(object) <- "ERVFL"
    return(object)
}
