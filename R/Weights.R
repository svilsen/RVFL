##########################################################################
####################### Ensemble weight estimation #######################
##########################################################################

weight_estimation_function <- function(pars, y, y_hat) {
    y_w <- y_hat %*% pars
    e <- y - y_w
    SSE <- c(t(e) %*% e)
    return(SSE)
}

weight_estimation_bound <- function(pars, y, y_hat) {
    return(sum(pars))
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
    y_hat <- predict(object, newdata = X_val, type = "full")
    
    w_0 <- runif(B) 
    w_0 <- w_0 / sum(w_0)
    w_hat <- solnp(
        pars = w_0, 
        fun = weight_estimation_function, 
        LB = rep(.Machine$double.eps, length(w_0)), 
        UB = rep(1L - .Machine$double.eps, length(w_0)), 
        eqfun = weight_estimation_bound,
        eqB = 1L,
        y = y_val, y_hat = y_hat,
        control = list(trace = trace, tol = 1e-12)
    )
    
    object$weights <- w_hat$pars
    return(object)
}
