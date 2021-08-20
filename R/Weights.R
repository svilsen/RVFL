##########################################################################
####################### Ensemble weight estimation #######################
##########################################################################

estimate_weights_stack <- function(C, b, B) {
    D <- t(C) %*% C + diag(1e-8, nrow = ncol(C), ncol = ncol(C))
    d <- t(C) %*% b
    A <- rbind(t(matrix(rep(1, B), ncol = 1)), diag(B), -diag(B))
    b <- c(1, rep(0, B), rep(-1, B))
    
    w <- solve.QP(D, d, t(A), b, meq = 1)$solution
    w[w < 0] <- 0
    w <- w / sum(w)
    
    return(w)
}

#' @title Set ensemble weights for an ERVFL-object.
#' 
#' @description Manually set ensemble weights for an ERVFL-object.
#' 
#' @param object An ERVFL-object.
#' @param weights A vector of ensemble weights.
#' 
#' @return An ERVFL-object.
#' 
#' @export
set_weights <- function(object, weights = NULL) {
    UseMethod("set_weights")
}

#' @rdname set_weights
#' @method set_weights ERVFL
#' 
#' @example inst/examples/sw_example.R
#'
#' @export
set_weights.ERVFL <- function(object, weights = NULL) {
    if (is.null(weights)) {
        warning("No weights defined, setting weights to uniform.")
        return(object)
    }
    
    if (length(weights) != length(object$weights)) {
        stop("The number of supplied weights have to be equal to the number of bootstrap samples.")
    }
    
    if (abs(sum(weights) - 1) > 1e-6) {
        stop("The weights have to sum to 1.")
    }
    
    if (any(weights > 1) || any(weights < 0)) {
        stop("All weights have to be between 0 and 1.")
    }
    
    object$weights <- weights
    return(object)
}

#' @title Estimate ensemble weights for an ERVFL-object.
#' 
#' @description Estimate ensemble weights for an ERVFL-object.
#' 
#' @param object An ERVFL-object.
#' @param X_val The validation feature set.
#' @param y_val The validation target set.
#' @param trace The trace of \link{solnp} are printed every '\code{trace}' number of iteration (default 0). 
#' 
#' @return An ERVFL-object.
#' 
#' @export
estimate_weights <- function(object, X_val = NULL, y_val = NULL, trace = 0) {
    UseMethod("estimate_weights")
}

#' @rdname estimate_weights
#' @method estimate_weights ERVFL
#' 
#' @example inst/examples/ew_example.R
#'
#' @export
estimate_weights.ERVFL <- function(object, X_val = NULL, y_val = NULL, trace = 0) {
    if (is.null(X_val) || is.null(y_val)) {
        warning("The validation-set was not properly specified, therefore, the training is used for weight estimation.")
        
        X_val <- object$data$X
        y_val <- object$data$y
    }
    
    B <- length(object$RVFLmodels)
    C <- predict(object, newdata = X_val, type = "full")
    
    object$weights <- estimate_weights_stack(C = C, b = y_val, B = B)
    return(object)
}
