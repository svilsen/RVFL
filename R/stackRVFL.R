#############################################################################
####################### Stackig ERVFL neural networks #######################
#############################################################################

#' @title Stacking random vector functional links
#' 
#' @description Use stacking to create ensemble random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of models in the stack.
#' @param folds The number of folds used to train the RVFL models. 
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param N_features The number of features randomly chosen in each iteration (default is \code{ceiling(ncol(X) / 3)}).
#' @param optimise TRUE/FALSE: Should the stacking weights be optimised (or should the stack just use the average)? 
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
stackRVFL <- function(X, y, N_hidden, B = 100, folds = 10, lambda = 0, N_features = NULL, optimise = TRUE, ...) {
    UseMethod("stackRVFL")
}

#' @rdname stackRVFL
#' @method stackRVFL default
#' 
#' @example inst/examples/stackrvfl_example.R
#' 
#' @export
stackRVFL.default <- function(X, y, N_hidden, B = 100, folds = 10, lambda = 0, N_features = NULL, optimise = TRUE, ...) {
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
        B <- 100
        
        warning(paste0("Note: 'B' was not supplied, 'B' was set to ", B, "."))
    }
    
    if (is.null(N_features)) {
        N_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    fold_index <- create_folds(X, folds)
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        object_b <- RVFL(X = X, y = y, N_hidden = N_hidden, lambda = lambda, N_features = N_features, ...)
        
        beta_b <- vector("list", folds)
        for (k in seq_len(folds)) {
            Xk <- matrix(X[-fold_index[[k]], ], ncol = ncol(X))
            yk <- matrix(y[-fold_index[[k]], ], ncol = ncol(y))
            
            Hk <- rvfl_forward(Xk, object_b$Weights$Hidden, object_b$activation, object_b$Bias$Hidden)
            Hk <- lapply(seq_along(Hk), function(i) matrix(Hk[[i]], ncol = N_hidden[i]))
            Hk <- do.call("cbind", Hk)
            
            ## Estimate parameters in output layer
            if (object_b$Bias$Output) {
                Hk <- cbind(1, Hk)
            }
            
            Ok <- Hk
            if (object_b$Combined) {
                Ok <- cbind(Xk, Hk)
            }
            
            beta_b[[k]] <- estimate_output_weights(Ok, yk, lambda)$beta
        }
        
        object_b$Weights$Output <- apply(do.call("cbind", beta_b), 1, mean)
        objects[[b]] <- object_b
    }
    
    ##
    if (optimise) {
        C <- do.call("cbind", lapply(objects, predict))
        w <- estimate_weights_stack(C = C, b = y, B = B)
    }
    else {
        w <- 1 / B
    }
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = objects, 
        weights = w, 
        method = "stacking"
    )  
    
    class(object) <- "ERVFL"
    return(object)
}
