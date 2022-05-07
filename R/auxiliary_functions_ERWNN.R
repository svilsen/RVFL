#########################################################################
####################### ERWNN neural networks AUX #######################
#########################################################################

#' @title Coefficients of the ERWNN-object
#' 
#' @param object An \link{ERWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The additional argument '\code{type}' is only used if '\code{method}' was \code{"bagging"}, in which case it can be supplied with values \code{"all"}, \code{"sd"}, and \code{"mean"} (default), returning the full list of coefficients for all bootstrap samples, the standard deviation of each coefficient across bootstrap samples, and the average value of each coefficient across bootstrap samples, respectively.
#' 
#' @return Depended on '\code{method}' and '\code{type}':
#' 
#' If '\code{method}' was \code{"bagging"}, the '\code{type}' yields the following results: 
#' \describe{
#'     \item{\code{"mean" (default):}}{A vector containing the average value of each parameter taken across the bootstrap samples.}
#'     \item{\code{"sd":}}{A vector containing the standard deviation of each parameter taken across the bootstrap samples.}
#'     \item{\code{"all":}}{A matrix where every column contains the parameters of the output-layer of corresponding boostrap sample.}
#' }
#' 
#' If '\code{method}' was \code{"boosting"} or \code{"stacking"}, a matrix is returned corresponding to '\code{type == "all"}'.
#' 
#' @rdname coef.ERWNN
#' @method coef ERWNN
#' @export
coef.ERWNN <- function(object, ...) {
    dots <- list(...)
    type <- dots$type
    if (is.null(type)) {
        type <- "mean"
    }
    else {
        type <- tolower(type)
    }
    
    if (object$method == "ed") {
        beta <- object$OutputWeights
    }
    else {
        B <- length(object$RWNNmodels)
        beta <- vector("list", B)
        for (b in seq_along(beta)) {
            beta[[b]] <- coef(object$RWNNmodels[[b]])
        }
        
        beta <- do.call("cbind", beta)
    }
    
    ##
    if (object$method %in% c("bagging")) {
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
            stop("The value of 'type' was not valid, see '?coef.ERWNN' for valid options of 'type'.")
        }
    }
    else {
        return(beta)
    }
}

#' @title Predicting targets of an ERWNN-object
#' 
#' @param object An \link{ERWNN-object}.
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
#' If '\code{method}' was \code{"bagging"}, \code{"stacking"}, \code{"ed"}, or \code{"resample"}, the '\code{type}' yields the following results: 
#' \describe{
#'     \item{\code{"mean" (default):}}{A vector containing the weighted (using the \code{weights} element of the \link{ERWNN-object}) sum each observation taken across the bootstrap samples.}
#'     \item{\code{"sd":}}{A vector containing the standard deviation of each prediction taken across the bootstrap samples.}
#'     \item{\code{"all":}}{A matrix where every column contains the predicted values corresponding to each of the boostrapped models.}
#' }
#' 
#' If '\code{method}' was \code{"boosting"}, a vector is returned each element being the sum of the boosted predictions.
#'
#' @rdname predict.ERWNN
#' @method predict ERWNN
#' @export
predict.ERWNN <- function(object, ...) {
    dots <- list(...)
    type <- dots$type
    if (is.null(type)) {
        type <- "mean"
    } else {
        type <- tolower(type)
    }
    
    if (is.null(dots$newdata)) {
        newdata <- object$data$X
    } else {
        if (is.null(object$formula)) {
            newdata <- as.matrix(dots$newdata)
        } else {
            #
            newdata <- model.matrix(object$formula, dots$newdata)
            keep <- which(colnames(newdata) != "(Intercept)")
            if (any(colnames(newdata) == "(Intercept)")) {
                newdata <- newdata[, keep]
            }
            
            newdata <- as.matrix(newdata, ncol = length(keep))
        }
        
        if (dim(newdata)[2] != (dim(object$RWNNmodels[[1]]$Weights$Hidden[[1]])[1] - as.numeric(object$RWNNmodels[[1]]$Bias$Hidden[1]))) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
    }
    
    ##
    B <- length(object$weights)
    if (object$method == "ed") {
        newH <- rwnn_forward(X = newdata, W = object$RWNNmodels$Weights$Hidden, activation = object$RWNNmodels$activation, bias = object$RWNNmodels$Bias$Hidden)
        newH <- lapply(seq_along(newH), function(i) matrix(newH[[i]], ncol = object$RWNNmodels$N_hidden[i]))
        newO <- lapply(seq_along(newH), function(i) cbind(1, newdata, newH[[i]]))
        y_new <- lapply(seq_along(newO), function(i) newO[[i]] %*% object$OutputWeights[[i]])
        
        y_new <- do.call("cbind", y_new)
    } else {
        y_new <- vector("list", B)
        for (b in seq_along(y_new)) {
            y_new[[b]] <- predict.RWNN(object = object$RWNNmodels[[b]], newdata = newdata)
        }
        
        y_new <- do.call("cbind", y_new)
    }
    
    ##
    if (object$method %in% c("bagging", "stacking", "ed", "resample")) {
        if (type %in% c("a", "all", "f", "full")) {
            return(y_new)
        } else if (type %in% c("m", "mean", "avg", "average")) {
            W <- matrix(rep(object$weights, dim(newdata)[1]), ncol = B, byrow = TRUE)
            y_new <- matrix(apply(y_new * W, 1, sum), ncol = 1)
            return(y_new)
        } else if (type %in% c("s", "sd", "standarddeviation")) {
            y_new <- matrix(apply(y_new, 1, sd), ncol = 1)
            return(y_new)
        } else {
            stop("The value of 'type' was not valid, see '?predict.ERWNN' for valid options of 'type'.")
        }
    } else {
        y_new <- matrix(apply(y_new, 1, sum), ncol = 1)
        return(y_new)
    }
}

#' @title Residuals of the ERWNN-object
#' 
#' @param object An \link{ERWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details No additional arguments are used in this instance.
#' 
#' @return A vector of raw residuals between the predicted (using \code{type = "mean"}) and observed targets.
#' 
#' @rdname residuals.ERWNN
#' @method residuals ERWNN
#' @export
residuals.ERWNN <- function(object, ...) {
    y_new <- predict(object, ...)
    
    r <- y_new - object$data$y
    return(r)
}

#' @title Diagnostic-plots of an ERWNN-object
#' 
#' @param x An \link{ERWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments used by the function are '\code{X_val}' and '\code{y_val}', i.e. the features and targets of the validation-set. These are helpful when analysing whether overfitting of model has occurred.  
#' 
#' @return NULL
#' 
#' @rdname plot.ERWNN
#' @method plot ERWNN
#'
#' @export
plot.ERWNN <- function(x, ...) {
    dots <- list(...)
    if (is.null(dots$X_val) && is.null(dots$y_val)) {
        X_val <- x$data$X
        y_val <- x$data$y
    } else if (is.null(dots$X_val) || is.null(dots$y_val)) {
        message("The testing-set was not properly specified, therefore, the training-set is used.")
        
        X_val <- x$data$X
        y_val <- x$data$y
    } else {
        X_val <- dots$X_val
        y_val <- dots$y_val
    }
    
    y_hat <- predict(x, newdata = X_val)
    
    if (is.null(dots$page)) {
        dev.hold()
        plot(y_hat ~ y_val, pch = 16, 
             xlab = "Observed targets", ylab = "Predicted targets")
        abline(0, 1, col = "dodgerblue", lty = "dashed", lwd = 2)
        dev.flush()
        
        readline(prompt = "Press [ENTER] for next plot...")
        dev.hold()
        plot(I(y_hat - y_val) ~ seq(length(y_val)), pch = 16,
             xlab = "Index", ylab = "Residual") 
        abline(0, 0, col = "dodgerblue", lty = "dashed", lwd = 2)
        dev.flush()
        
        readline(prompt = "Press [ENTER] for next plot...")
        dev.hold()
        plot(x$weights ~ seq(length(x$weights)), pch = 16,
             xlab = "Bootstrap index", ylab = "Weights") 
        dev.flush()
    } else if (dots$page == 1) {
        dev.hold()
        plot(y_hat ~ y_val, pch = 16, 
             xlab = "Observed targets", ylab = "Predicted targets")
        abline(0, 1, col = "dodgerblue", lty = "dashed", lwd = 2)
        dev.flush()
    } else if (dots$page == 2) {
        dev.hold()
        plot(I(y_hat - y_val) ~ seq(length(y_val)), pch = 16,
             xlab = "Index", ylab = "Residual") 
        abline(0, 0, col = "dodgerblue", lty = "dashed", lwd = 2)
        dev.flush()
    } else if (dots$page == 3) {
        dev.hold()
        plot(x$weights ~ seq(length(x$weights)), pch = 16,
             xlab = "Bootstrap index", ylab = "Weights") 
        dev.flush()
    } else {
        stop("Invalid choice of 'page', it has to take the values 1, 2, 3, or NULL.")
    }
    
    return(invisible(NULL))
}

estimate_weights_stack <- function(C, b, B) {
    # Creating matricies for QP optimisation problem.
    # NB: diagonal matrix is added to ensure the matrix is invertible.
    D <- t(C) %*% C + diag(1e-8, nrow = ncol(C), ncol = ncol(C))
    d <- t(C) %*% b
    A <- rbind(t(matrix(rep(1, B), ncol = 1)), diag(B), -diag(B))
    b <- c(1, rep(0, B), rep(-1, B))
    
    # Solution to QP optimisation problem
    w <- solve.QP(D, d, t(A), b, meq = 1)$solution
    
    # Ensure all weights are >= 1e-8 (some may not be due to machine precision)
    w[w < 1e-8] <- 1e-8
    w <- w / sum(w)
    
    return(w)
}

#' @title Set ensemble weights for an ERWNN-object
#' 
#' @description Manually set ensemble weights for an \link{ERWNN-object}.
#' 
#' @param object An \link{ERWNN-object}.
#' @param weights A vector of ensemble weights.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
set_weights <- function(object, weights = NULL) {
    UseMethod("set_weights")
}

#' @rdname set_weights
#' @method set_weights ERWNN
#' 
#' @example inst/examples/sw_example.R
#'
#' @export
set_weights.ERWNN <- function(object, weights = NULL) {
    if (is.null(weights)) {
        warning("No weights defined, setting weights to uniform.")
        return(object)
    }
    
    if (length(weights) != length(object$weights)) {
        stop("The number of supplied weights have to be equal to the number of ensemble weights.")
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

#' @title Estimate ensemble weights for an ERWNN-object
#' 
#' @description Estimate ensemble weights for an \link{ERWNN-object}.
#' 
#' @param object An \link{ERWNN-object}.
#' @param X_val The validation feature set.
#' @param y_val The validation target set.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
estimate_weights <- function(object, X_val = NULL, y_val = NULL) {
    UseMethod("estimate_weights")
}

#' @rdname estimate_weights
#' @method estimate_weights ERWNN
#' 
#' @example inst/examples/ew_example.R
#'
#' @export
estimate_weights.ERWNN <- function(object, X_val = NULL, y_val = NULL) {
    if (is.null(X_val) || is.null(y_val)) {
        warning("The validation-set was not properly specified, therefore, the training-set is used for weight estimation.")
        
        X_val <- object$data$X
        y_val <- object$data$y
    } 
    
    B <- length(object$RWNNmodels)
    C <- predict(object, newdata = X_val, type = "full")
    
    object$weights <- estimate_weights_stack(C = C, b = y_val, B = B)
    return(object)
}
