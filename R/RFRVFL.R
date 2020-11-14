
#' @title Random forest random vector functional link
#' 
#' @description Set-up and estimate weights the ensemble random forest random vector functional link neural network.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of bootstrap samples. 
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An RFRVFL-object containing the random and fitted weights of all bootstrapped RVFL-model.
#' 
#' @export
RFRVFL <- function(X, y, N_hidden, B = NULL, ...) {
    UseMethod("RFRVFL")
}

#' @rdname RFRVFL
#' @method RFRVFL default
#' 
#' @example inst/examples/rfrvfl_example.R
#' 
#' @export
RFRVFL.default <- function(X, y, N_hidden, B, ...) {
    ##
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        indices_b <- sample(nrow(X), nrow(X), replace = TRUE)
        X_b <- matrix(X[indices_b, ], ncol = ncol(X))
        y_b <- matrix(y[indices_b], ncol = ncol(y))
        
        objects[[b]] <- RVFL(X = X_b, y = y_b, N_hidden = N_hidden, ...)
    }
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = objects, 
        weights = rep(1L / B, B)
    )  
    
    class(object) <- "RFRVFL"
    return(object)
}

#' @title Coefficients of the RFRVFL object.
#' 
#' @param object An RFRVFL-object.
#' @param ... Additional arguments.
#' 
#' @rdname coef
#' @method coef RFRVFL
#' @export
coef.RFRVFL <- function(object, ...) {
    dots <- list(...)
    
    B <- length(object$RVFLmodels)
    beta <- vector("list", B)
    for (b in seq_along(beta)) {
        beta[[b]] <- coef(object$RVFLmodels[[b]])
    }
    
    beta <- do.call("cbind", beta)
    
    ##
    if (is.null(dots$type)) {
        beta <- matrix(apply(beta, 1, mean), ncol = 1)
        return(beta)
    }
    else {
        type <- tolower(dots$type)
        if (type %in% c("a", "all", "f", "full")) {
            return(beta)
        }
        else if (type %in% c("sd", "standarddeviation")) {
            beta <- matrix(apply(beta, 1, sd), ncol = 1)
            return(beta)
        }
    }
}

#' @title Predicting targets of an RFRVFL object.
#' 
#' @param object An RFRVFL-object.
#' @param ... Additional arguments.
#' 
#' @rdname predict
#' @method predict RFRVFL
#' @export
predict.RFRVFL <- function(object, ...) {
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
    
    ##
    B <- length(object$RVFLmodels)
    newy <- vector("list", B)
    for (b in seq_along(newy)) {
        newy[[b]] <- RFRVFL:::predict.RVFL(object = object$RVFLmodels[[b]], newdata = newdata)
    }
    
    newy <- do.call("cbind", newy)
    
    ##
    if (is.null(dots$type)) {
        W <- matrix(rep(object$weights, dim(newdata)[1]), ncol = B, byrow = TRUE)
        newy <- matrix(apply(newy * W, 1, sum), ncol = 1)
        return(newy)
    }
    else {
        type <- tolower(dots$type)
        if (type %in% c("a", "all", "f", "full")) {
            return(newy)
        }
        else if (type %in% c("sd", "standarddeviation")) {
            newy <- matrix(apply(newy, 1, sd), ncol = 1)
            return(newy)
        }
    }
}

#' @title Residuals of the RFRVFL object.
#' 
#' @param object An RFRVFL-object.
#' @param ... Additional arguments.
#' 
#' @rdname residuals
#' @method residuals RFRVFL
#' @export
residuals.RFRVFL <- function(object, ...) {
    dots <- list(...)
    newy <- predict(object)
    
    r <- newy - object$data$y
    return(r)
}


#' @title Set ensemble weights for an RFRVFL-object.
#' 
#' @description Manually set ensemble weights for an RFRVFL-object.
#' 
#' @param object An RFRVFL-object.
#' @param weights A vector of ensemble weights.
#' 
#' @return An RFRVFL-object.
#' 
#' @export
set_weights <- function(object, weights = NULL) {
    UseMethod("set_weights")
}

#' @title Set ensemble weights for an RFRVFL-object.
#' 
#' @param object An RFRVFL-object.
#' @param weights A vector of ensemble weights.
#' 
#' @rdname set_weights
#' @method set_weights RFRVFL
#' @export
set_weights.RFRVFL <- function(object, weights = NULL) {
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


weight_estimation_function <- function(pars, y, y_hat) {
    y_w <- y_hat %*% pars
    e <- y - y_w
    SSE <- c(t(e) %*% e)
    return(SSE)
}

weight_estimation_bound <- function(pars, y, y_hat) {
    return(sum(pars))
}

#' @title Estimate ensemble weights for an RFRVFL-object.
#' 
#' @description Estimate ensemble weights for an RFRVFL-object.
#' 
#' @param object An RFRVFL-object.
#' @param validation_X The validation feature set.
#' @param validation_y The validation target set.
#' 
#' @return An RFRVFL-object.
#' 
#' @export
estimate_weights <- function(object, validation_X = NULL, validation_y = NULL) {
    UseMethod("estimate_weights")
}

#' @title Estimate ensemble weights for an RFRVFL-object.
#' 
#' @param object An RFRVFL-object.
#' @param validation_X The validation feature set.
#' @param validation_y The validation target set.
#' 
#' @rdname estimate_weights
#' @method estimate_weights RFRVFL
#' @export
estimate_weights.RFRVFL <- function(object, validation_X = NULL, validation_y = NULL) {
    if (is.null(validation_X) || is.null(validation_y)) {
        warning("The validation-set was not properly specified, therefore, the training is used for weight estimation. This is not ideal as it will lead to overestimation.")
        
        validation_X <- object$data$X
        validation_y <- object$data$y
    }
    
    B <- length(object$RVFLmodels)
    y_hat <- predict(object, newdata = validation_X, type = "full")
    
    w_0 <- runif(B) 
    w_0 <- w_0 / sum(w_0)
    w_hat <- Rsolnp::solnp(
        pars = w_0, 
        fun = weight_estimation_function, 
        LB = rep(.Machine$double.eps, length(w_0)), 
        UB = rep(1L - .Machine$double.eps, length(w_0)), 
        eqfun = weight_estimation_bound,
        eqB = 1L,
        y = validation_y, y_hat = y_hat,
        control = list(trace = FALSE, tol = 1e-16)
    )
    
    object$weights <- w_hat$pars
    return(object)
}
