#############################################################################
####################### Boosting ERVFL neural network #######################
#############################################################################

#' @title Boosting random vector functional links
#' 
#' @description Use gradient boosting to create ensemble random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of levels used in the boosting tree.
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param epsilon The learning rate.
#' @param N_features The number of features randomly chosen in each iteration (default is \code{ceiling(ncol(X) / 3)}).
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An ERVFL-object containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list of \link{RVFL}-objects.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method ('boosting' in this case)}
#' }
#' 
#' @export
boostRVFL <- function(X, y, N_hidden, B = 10, lambda = 0, epsilon = 1, N_features = NULL, ...) {
    UseMethod("boostRVFL")
}

#' @rdname boostRVFL
#' @method boostRVFL default
#' 
#' @example inst/examples/boostrvfl_example.R
#' 
#' @export
boostRVFL.default <- function(X, y, N_hidden, B = 10, lambda = 0, epsilon = 1, N_features = NULL, ...) {
    ## Checks
    # Data
    if (!is.matrix(X)) {
        warning("'X' has to be a matrix... trying to cast 'X' as a matrix.")
        X <- as.matrix(X)
    }
    
    if (!is.matrix(y)) {
        warning("'y' has to be a matrix... trying to cast 'y' as a matrix.")
        y <- as.matrix(y)
    }
    
    if (dim(y)[2] != 1) {
        warning("More than a single column was detected in 'y', only the first column is used in the model.")
        y <- matrix(y[, 1], ncol = 1)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    if (is.null(B)) {
        B <- 10
        
        warning(paste0("Note: 'B' was not supplied, 'B' was set to ", B, "."))
    }
    
    if (is.null(epsilon)) {
        epsilon <- 1
        warning("Note: 'epsilon' was not supplied and set to 1.")
    }
    else if (epsilon > 1) {
        epsilon <- 1
        warning("'epsilon' has to be a number between 0 and 1.")
    }
    else if (epsilon < 0) {
        epsilon <- 0
        warning("'epsilon' has to be a number between 0 and 1.")
    }
    
    if (is.null(N_features)) {
        N_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        X_b <- X
        if (b == 1) {
            y_b <- y
        }
        else {
            y_b <- y_b - epsilon * predict(objects[[b - 1]])
        }
        
        objects[[b]] <- RVFL(X = X_b, y = y_b, N_hidden = N_hidden, lambda = lambda, N_features = N_features, ...)
    }
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = objects, 
        weights = rep(1L / B, B), 
        method = "boosting"
    )  
    
    class(object) <- "ERVFL"
    return(object)
}

