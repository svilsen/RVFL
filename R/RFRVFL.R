
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
        RVFLmodels = objects
    )  
    
    class(object) <- "RFRVFL"
    return(object)
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
        if (dim(newdata)[2] != dim(object$data$X)[2]) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
    }
    
    ##
    B <- length(object$RVFLmodels)
    newy <- vector("list", B)
    for (b in seq_along(newy)) {
        newy[[b]] <- predict(object$RVFLmodels[[b]], newdata = newdata)
    }
    
    newy <- do.call("cbind", newy)
    
    ##
    if (is.null(dots$type)) {
        newy <- matrix(apply(newy, 1, mean), ncol = 1)
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

