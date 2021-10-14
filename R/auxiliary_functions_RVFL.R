########################################################################
####################### RVFL neural networks AUX #######################
########################################################################

#' @title Coefficients of the RVFL-object.
#' 
#' @param object An \link{RVFL-object}.
#' @param ... Additional arguments.
#' 
#' @details No additional arguments are used in this instance.
#' 
#' @return The estimated weights of the output-layer.
#' 
#' @rdname coef.RVFL
#' @method coef RVFL
#' @export
coef.RVFL <- function(object, ...) {
    return(object$Weights$Output)
}

#' @title Predicting targets of an RVFL-object.
#' 
#' @param object An \link{RVFL-object}.
#' @param ... Additional arguments.
#' 
#' @details The only additional argument used by the function is '\code{newdata}', which expects a matrix with the same number of features (columns) as in the original data.
#' 
#' @return A vector of predicted targets.
#' 
#' @rdname predict.RVFL
#' @method predict RVFL
#' @export
predict.RVFL <- function(object, ...) {
    dots <- list(...)
    
    if (is.null(dots$newdata)) {
        if (is.null(object$data)) {
            stop("The RVFL-object does not contain any data: Either supply 'newdata', or re-create object with 'include_data = TRUE' (default).")
        }
        
        newdata <- object$data$X
    } else {
        if (dim(dots$newdata)[2] != (dim(object$Weights$Hidden[[1]])[1] - as.numeric(object$Bias$Hidden[1]))) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
        
        newdata <- dots$newdata 
    }
    
    newH <- rvfl_forward(
        X = newdata, 
        W = object$Weights$Hidden, 
        activation = object$activation,
        bias = object$Bias$Hidden
    )
    
    newH <- lapply(seq_along(newH), function(i) matrix(newH[[i]], ncol = object$N_hidden[i]))
    newH <- do.call("cbind", newH)
    
    ## Estimate parameters in output layer
    if (object$Bias$Output) {
        newH <- cbind(1, newH)
    }
    
    newO <- newH
    if (object$Combined) {
        newO <- cbind(newH, newdata)
    }
    
    newy <- newO %*% object$Weights$Output
    return(newy)
}

#' @title Residuals of the RVFL-object.
#' 
#' @param object An \link{RVFL-object}.
#' @param ... Additional arguments.
#' 
#' @details Besides the arguments passed to the '\code{predict}' function, the argument '\code{type}' can be supplied defining the type of residual returned by the function. Currently only \code{"rs"} (standardised residuals), and \code{"raw"} (default) are implemented.
#'
#' @return A vector of residuals of the desired '\code{type}' (see details). 
#'
#' @rdname residuals.RVFL
#' @method residuals RVFL
#' @export
residuals.RVFL <- function(object, ...) {
    dots <- list(...)
    type <- dots$type
    if (is.null(type)) {
        type <- "raw"
    }
    
    newy <- predict.RVFL(object, ...)
    
    r <- newy - object$data$y
    if (tolower(type) %in% c("standard", "standardised", "rs")) {        
        r <- r / object$Sigma$Output
    }
    
    return(r)
}

#' @title Diagnostic-plots of an RVFL-object.
#' 
#' @param x An \link{RVFL-object}.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments used by the function are '\code{X_val}' and '\code{y_val}', i.e. the features and targets of the validation-set. These are helpful when analysing whether overfitting of model has occured.  
#' 
#' @rdname plot.RVFL
#' @method plot RVFL
#'
#' @return NULL
#' 
#' @export
plot.RVFL <- function(x, ...) {
    dots <- list(...)
    if (is.null(dots$X_val) || is.null(dots$y_val)) {
        if (is.null(x$data)) {
            stop("The RVFL-object does not contain any data: Either supply 'X_val' and 'y_val', or re-create RVFL-object with 'include_data = TRUE' (default).")
        }
        
        X_val <- x$data$X
        y_val <- x$data$y
    }
    else {
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
    }
    else if (dots$page == 1) {
        dev.hold()
        plot(y_hat ~ y_val, pch = 16, 
             xlab = "Observed targets", ylab = "Predicted targets")
        abline(0, 1, col = "dodgerblue", lty = "dashed", lwd = 2)
        dev.flush()
    }
    else if (dots$page == 2) {
        dev.hold()
        plot(I(y_hat - y_val) ~ seq(length(y_val)), pch = 16,
             xlab = "Index", ylab = "Residual") 
        abline(0, 0, col = "dodgerblue", lty = "dashed", lwd = 2)
        dev.flush()
    }
    else {
        stop("Invalid choice of 'page', it has to take the values 1, 2, or NULL.")
    }
    
    return(invisible(NULL))
}
