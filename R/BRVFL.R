###############################################################################
####################### An ensemble RVFL neural network #######################
###############################################################################

#' @title Bagging random vector functional link
#' 
#' @description Set-up and estimate weights the ensemble random vector functional link neural network.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of bootstrap samples. 
#' @param ... Additional arguments. 
#' 
#' @details The additional arguments are all passed to the \link{control_RVFL} function.
#' 
#' @return A BRVFL-object containing the random and fitted weights of all bootstrapped \link{RVFL}-model. A BRVFL-object contains the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list of \link{RVFL}-objects.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#' }
#' 
#' @export
BRVFL <- function(X, y, N_hidden, B = NULL, ...) {
    UseMethod("BRVFL")
}

#' @rdname BRVFL
#' @method BRVFL default
#' 
#' @example inst/examples/brvfl_example.R
#' 
#' @export
BRVFL.default <- function(X, y, N_hidden, B, ...) {
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
    
    class(object) <- "BRVFL"
    return(object)
}

#' @title Coefficients of the BRVFL object.
#' 
#' @param object A BRVFL-object.
#' @param ... Additional arguments.
#' 
#' @details The additional argument '\code{type}' can be supplied with values \code{"all"}, \code{"sd"}, and \code{"mean"} (default), returning the full list of coefficients for all bootstrap samples, the standard deviation of each coefficient across bootstrap samples, and the average value of each coefficient across bootstrap samples, respectively.
#' 
#' @return Depended on \code{type}: 
#' \describe{
#'     \item{\code{"all"}}{A matrix where every column contains the parameters of the output-layer of corresponding boostrap sample.}
#'     \item{\code{"sd"}}{A vector containing the standard deviation of each parameter taken across the bootstrap samples.}
#'     \item{\code{"mean"}}{A vector containing the average value of each parameter taken across the bootstrap samples.}
#' }
#' 
#' @rdname coef.BRVFL
#' @method coef BRVFL
#' @export
coef.BRVFL <- function(object, ...) {
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

#' @title Predicting targets of an BRVFL object.
#' 
#' @param object A BRVFL-object.
#' @param ... Additional arguments.
#' 
#' @details The additional argument '\code{newdata}' and '\code{type}' can be specified:
#' \describe{
#'   \item{\code{newdata}}{Expects a matrix the same number of features (columns) as in the original data.}
#'   \item{\code{type}}{Takes values \code{"all"}, \code{"sd"}, and \code{"mean"} (default), returning a full matrix of predictions for all bootstrap samples, the standard deviation of each predicted observation across bootstrap samples, and the average value of each prediction across the bootstrap samples, respectively.}
#' }
#'
#' @return Depended on \code{type}: 
#' \describe{
#'     \item{\code{"all"}}{A matrix where every column contains the predicted values corresponding to each of the boostrapped models.}
#'     \item{\code{"sd"}}{A vector containing the standard deviation of each prediction taken across the bootstrap samples.}
#'     \item{\code{"mean"}}{A vector containing the weighted (using the \code{weights} element of the \link{BRVFL}-object) sum each observation taken across the bootstrap samples.}
#' }
#'
#' @rdname predict.BRVFL
#' @method predict BRVFL
#' @export
predict.BRVFL <- function(object, ...) {
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
        newy[[b]] <- predict.RVFL(object = object$RVFLmodels[[b]], newdata = newdata)
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

#' @title Residuals of the BRVFL object.
#' 
#' @param object A BRVFL-object.
#' @param ... Additional arguments.
#' 
#' @details No additional arguments are used in this instance.
#' 
#' @return A vector of raw residuals between the predicted (using \code{type = "mean"}) and observed targets.
#' 
#' @rdname residuals.BRVFL
#' @method residuals BRVFL
#' @export
residuals.BRVFL <- function(object, ...) {
    dots <- list(...)
    newy <- predict(object)
    
    r <- newy - object$data$y
    return(r)
}


#' @title Set ensemble weights for an BRVFL-object.
#' 
#' @description Manually set ensemble weights for an BRVFL-object.
#' 
#' @param object An BRVFL-object.
#' @param weights A vector of ensemble weights.
#' 
#' @return A \link{BRVFL}-object.
#' 
#' @export
set_weights <- function(object, weights = NULL) {
    UseMethod("set_weights")
}

#' @rdname set_weights
#' @method set_weights BRVFL
#' 
#' @example inst/examples/sw_example.R
#'
#' @export
set_weights.BRVFL <- function(object, weights = NULL) {
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

#' @title Estimate ensemble weights for an BRVFL-object.
#' 
#' @description Estimate ensemble weights for an BRVFL-object.
#' 
#' @param object A BRVFL-object.
#' @param validation_X The validation feature set.
#' @param validation_y The validation target set.
#' @param trace The trace of \link{solnp} are printed every 'trace' number of iteration (default 0). 
#' 
#' @return A \link{BRVFL}-object.
#' 
#' @export
estimate_weights <- function(object, validation_X = NULL, validation_y = NULL, trace = 0) {
    UseMethod("estimate_weights")
}

#' @rdname estimate_weights
#' @method estimate_weights BRVFL
#' 
#' @example inst/examples/ew_example.R
#'
#' @export
estimate_weights.BRVFL <- function(object, validation_X = NULL, validation_y = NULL, trace = 0) {
    if (is.null(validation_X) || is.null(validation_y)) {
        warning("The validation-set was not properly specified, therefore, the training is used for weight estimation.")
        
        validation_X <- object$data$X
        validation_y <- object$data$y
    }
    
    B <- length(object$RVFLmodels)
    y_hat <- predict(object, newdata = validation_X, type = "full")
    
    w_0 <- runif(B) 
    w_0 <- w_0 / sum(w_0)
    w_hat <- solnp(
        pars = w_0, 
        fun = weight_estimation_function, 
        LB = rep(.Machine$double.eps, length(w_0)), 
        UB = rep(1L - .Machine$double.eps, length(w_0)), 
        eqfun = weight_estimation_bound,
        eqB = 1L,
        y = validation_y, y_hat = y_hat,
        control = list(trace = trace, tol = 1e-12)
    )
    
    object$weights <- w_hat$pars
    return(object)
}

#' @title Diagnostic-plots of an BRVFL-object.
#' 
#' @param x A BRVFL-object.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments used by the function are '\code{testing_X}' and '\code{testing_y}', i.e. the features and targets of the testing-set. These are helpful when analysing whether overfitting of model has occured.  
#' 
#' @return NULL
#' 
#' @rdname plot.BRVFL
#' @method plot BRVFL
#'
#' @export
plot.BRVFL <- function(x, ...) {
    dots <- list(...)
    if (is.null(dots$testing_X) || is.null(dots$testing_y)) {
        message("The testing-set was not properly specified, therefore, the training-set is used.")
        
        testing_X <- x$data$X
        testing_y <- x$data$y
    }
    else {
        testing_X <- dots$testing_X
        testing_y <- dots$testing_y
    }
    
    y_hat <- predict(x, newdata = testing_X)
    
    dev.hold()
    plot(y_hat ~ testing_y, pch = 16, 
         xlab = "Observed targets", ylab = "Predicted targets")
    abline(0, 1, col = "dodgerblue", lty = "dashed", lwd = 2)
    dev.flush()
    
    readline(prompt = "Press [ENTER] for next plot...")
    dev.hold()
    plot(I(y_hat - testing_y) ~ seq(length(testing_y)), pch = 16,
         xlab = "Index", ylab = "Residual") 
    abline(0, 0, col = "dodgerblue", lty = "dashed", lwd = 2)
    dev.flush()
    
    readline(prompt = "Press [ENTER] for next plot...")
    dev.hold()
    plot(x$weights ~ seq(length(x$weights)), pch = 16,
         xlab = "Bootstrap index", ylab = "Weights") 
    dev.flush()
    
    return(invisible(NULL))
}




