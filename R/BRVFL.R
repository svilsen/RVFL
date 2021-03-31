###############################################################################
####################### An ensemble RVFL neural network #######################
###############################################################################

#' @title Bagging and boosting random vector functional link
#' 
#' @description Set-up and estimate weights the ensemble random vector functional link neural network, while either bagging or boosting for added stability of the model.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B If '\code{method}' is \code{"bagging"}, then it is the number of bootstrap samples. If \code{method} is \code{"boosting"}, it is the number of levels used when boosting the model.
#' @param method A string specifying whether \code{"bagging"} (default) or \code{"boosting"} should be performed on the RVFL.
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
BRVFL <- function(X, y, N_hidden, B = NULL, method = NULL, ...) {
    UseMethod("BRVFL")
}

#' @rdname BRVFL
#' @method BRVFL default
#' 
#' @example inst/examples/brvfl_example.R
#' 
#' @export
BRVFL.default <- function(X, y, N_hidden, B, method, ...) {
    ## Checks
    # Data
    if (!is.matrix(X)) {
        stop("'X' has to be a matrix.")
    }
    
    if (!is.matrix(y)) {
        stop("'y' has to be a matrix.")
    }
    
    if (dim(y)[2] != 1) {
        warning("Note: More than a single column was detected in 'y', only the first column is used in the model.")
        y <- matrix(y[, 1], ncol = 1)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    # Parameters
    if (is.null(method)) {
        warning("Note: 'method' was not set, it will be set to 'bagging'.")
        method <- "bagging"
    }
    
    method <- tolower(method)
    if (method %in% c("bg", "bag", "bagging")) {
        method <- "bagging"
    }
    else if (method %in% c("bs", "bt", "boost", "boosting")) {
        method <- "boosting"
    }
    else {
        stop("'method' has to be set to either 'bagging' or 'boosting'.")
    }
    
    if (is.null("B")) {
        if (method == "bagging") {
            B <- 100
        }
        else {
            B <- 10
        }
        
        warning(paste0("Note: 'B' was not supplied -- due to the choice of 'method', 'B' was set to ", B, "."))
    }
    
    ##
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        if (method == "bagging") {
            indices_b <- sample(nrow(X), nrow(X), replace = TRUE)
            X_b <- matrix(X[indices_b, ], ncol = ncol(X))
            y_b <- matrix(y[indices_b], ncol = ncol(y))    
        }
        else {
            X_b <- X
            if (b == 1) {
                y_b <- y
            }
            else {
                y_b <- residuals(objects[[b - 1]])
            }
        }
        
        objects[[b]] <- RVFL(X = X_b, y = y_b, N_hidden = N_hidden, ...)
    }
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = objects, 
        weights = rep(1L / B, B), 
        method = method
    )  
    
    class(object) <- "BRVFL"
    return(object)
}

#' @title Coefficients of the BRVFL object.
#' 
#' @param object A BRVFL-object.
#' @param ... Additional arguments.
#' 
#' @details The additional argument '\code{type}' is only used if '\code{method}' was set to \code{"bagging"}, in which case it can be supplied with values \code{"all"}, \code{"sd"}, and \code{"mean"} (default), returning the full list of coefficients for all bootstrap samples, the standard deviation of each coefficient across bootstrap samples, and the average value of each coefficient across bootstrap samples, respectively.
#' 
#' @return Depended on '\code{method}' and '\code{type}':
#' 
#' If '\code{method}' was set to \code{"bagging"}, the '\code{type}' yields the following results: 
#' \describe{
#'     \item{\code{"all"}}{A matrix where every column contains the parameters of the output-layer of corresponding boostrap sample.}
#'     \item{\code{"sd"}}{A vector containing the standard deviation of each parameter taken across the bootstrap samples.}
#'     \item{\code{"mean"}}{A vector containing the average value of each parameter taken across the bootstrap samples.}
#' }
#' 
#' If '\code{method}' was set to \code{"boosting"}, a matrix is returned corresponding to '\code{type == "all"}'.
#' 
#' @rdname coef.BRVFL
#' @method coef BRVFL
#' @export
coef.BRVFL <- function(object, ...) {
    dots <- list(...)
    type <- dots$type
    if (is.null(type)) {
        type <- "mean"
    }
    else {
        type <- tolower(type)
    }
    
    B <- length(object$RVFLmodels)
    beta <- vector("list", B)
    for (b in seq_along(beta)) {
        beta[[b]] <- coef(object$RVFLmodels[[b]])
    }
    
    beta <- do.call("cbind", beta)
    
    ##
    if (object$method == "bagging") {
        if (type %in% c("a", "all", "f", "full")) {
            return(beta)
        }
        else if (type %in% c("m", "mean", "avg", "average")) {
            beta <- matrix(apply(beta, 1, mean), ncol = 1)
            return(beta)
        }            
        else if (type %in% c("s", "sd", "standarddeviation")) {
            beta <- matrix(apply(beta, 1, sd), ncol = 1)
            return(beta)
        }
        else {
            stop("The passed value of 'type' was not valid. See '?coef.BRVFL' for valid options of 'type'.")
        }
    }
    else {
        return(beta)
    }
}

#' @title Predicting targets of an BRVFL object.
#' 
#' @param object A BRVFL-object.
#' @param ... Additional arguments.
#' 
#' @details The additional argument '\code{newdata}' and '\code{type}' can be specified, as follows:
#' \describe{
#'   \item{\code{newdata}}{Expects a matrix the same number of features (columns) as in the original data.}
#'   \item{\code{type}}{Is only used of '\code{method}' was set to \code{"bagging"}, in which case it takes values \code{"all"}, \code{"sd"}, and \code{"mean"} (default), returning a full matrix of predictions for all bootstrap samples, the standard deviation of each predicted observation across bootstrap samples, and the average value of each prediction across the bootstrap samples, respectively.}
#' }
#'
#' @return Depended on '\code{method}' and '\code{type}'. 
#' 
#' If '\code{method}' was set to \code{"bagging"}, the '\code{type}' yields the following results: 
#' \describe{
#'     \item{\code{"all"}}{A matrix where every column contains the predicted values corresponding to each of the boostrapped models.}
#'     \item{\code{"sd"}}{A vector containing the standard deviation of each prediction taken across the bootstrap samples.}
#'     \item{\code{"mean"}}{A vector containing the weighted (using the \code{weights} element of the \link{BRVFL}-object) sum each observation taken across the bootstrap samples.}
#' }
#' 
#' If '\code{method}' was set to \code{"boosting"}, a vector is returned each element being the sum of the boosted predictions.
#'
#' @rdname predict.BRVFL
#' @method predict BRVFL
#' @export
predict.BRVFL <- function(object, ...) {
    dots <- list(...)
    type <- dots$type
    if (is.null(type)) {
        type <- "mean"
    }
    else {
        type <- tolower(type)
    }
    
    if (is.null(dots$newdata)) {
        newdata <- object$data$X
    }
    else {
        if (dim(dots$newdata)[2] != dim(object$data$X)[2]) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
        
        newdata <- dots$newdata 
        
        if (!is.matrix(newdata)) {
            newdata <- as.matrix(newdata)
        }
    }
    
    ##
    B <- length(object$RVFLmodels)
    newy <- vector("list", B)
    for (b in seq_along(newy)) {
        newy[[b]] <- predict.RVFL(object = object$RVFLmodels[[b]], newdata = newdata)
    }
    
    newy <- do.call("cbind", newy)
    
    ##
    if (object$method == "bagging") {
        if (type %in% c("a", "all", "f", "full")) {
            return(newy)
        }
        else if (type %in% c("m", "mean", "avg", "average")) {
            W <- matrix(rep(object$weights, dim(newdata)[1]), ncol = B, byrow = TRUE)
            newy <- matrix(apply(newy * W, 1, sum), ncol = 1)
            return(newy)
        }
        else if (type %in% c("s", "sd", "standarddeviation")) {
            newy <- matrix(apply(newy, 1, sd), ncol = 1)
            return(newy)
        }
        else {
            stop("The passed value of 'type' was not valid. See '?coef.BRVFL' for valid options of 'type'.")
        }
    }
    else {
        newy <- matrix(apply(newy, 1, sum), ncol = 1)
        return(newy)
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
#' @param trace The trace of \link{solnp} are printed every '\code{trace}' number of iteration (default 0). 
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
#' @details The additional arguments used by the function are '\code{testing_X}' and '\code{testing_y}', i.e. the features and targets of the testing-set. These are helpful when analysing whether overfitting of model has occurred.  
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




