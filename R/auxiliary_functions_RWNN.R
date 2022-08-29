########################################################################
####################### RWNN neural networks AUX #######################
########################################################################

#' @title Coefficients of the RWNN-object
#' 
#' @param object An \link{RWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details No additional arguments are used in this instance.
#' 
#' @return The estimated weights of the output-layer.
#' 
#' @rdname coef.RWNN
#' @method coef RWNN
#' @export
coef.RWNN <- function(object, ...) {
    return(object$Weights$Output)
}

#' @title Predicting targets of an RWNN-object
#' 
#' @param object An \link{RWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The only additional argument used by the function is '\code{newdata}', which expects a matrix with the same number of features (columns) as in the original data.
#' 
#' @return A vector of predicted targets.
#' 
#' @rdname predict.RWNN
#' @method predict RWNN
#' @export
predict.RWNN <- function(object, ...) {
    dots <- list(...)
    
    if (is.null(dots$newdata)) {
        if (is.null(object$data)) {
            stop("The RWNN-object does not contain any data: Either supply 'newdata', or re-create object with 'include_data = TRUE' (default).")
        }
        
        newdata <- object$data$X        
    } else {
        if (is.null(object$formula)) {
            newdata <- as.matrix(dots$newdata)
        }
        else {
            #
            formula <- as.formula(object$formula)
            formula <- strip_terms(delete.response(terms(formula)))
            
            #
            newdata <- dots$newdata
            if (!is.data.frame(newdata)) {
                newdata <- as.data.frame(newdata)
            }
            
            #
            newdata <- model.matrix(formula, newdata)
            keep <- which(colnames(newdata) != "(Intercept)")
            if (any(colnames(newdata) == "(Intercept)")) {
                newdata <- newdata[, keep]
            }
            
            newdata <- as.matrix(newdata, ncol = length(keep))
        }
        
        if (dim(newdata)[2] != (dim(object$Weights$Hidden[[1]])[1] - as.numeric(object$Bias$Hidden[1]))) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
    }
    
    newH <- rwnn_forward(
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

#' @title Residuals of the RWNN-object
#' 
#' @param object An \link{RWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details Besides the arguments passed to the '\code{predict}' function, the argument '\code{type}' can be supplied defining the type of residual returned by the function. Currently only \code{"rs"} (standardised residuals), and \code{"raw"} (default) are implemented.
#'
#' @return A vector of residuals of the desired '\code{type}' (see details). 
#'
#' @rdname residuals.RWNN
#' @method residuals RWNN
#' @export
residuals.RWNN <- function(object, ...) {
    dots <- list(...)
    type <- dots$type
    if (is.null(type)) {
        type <- "raw"
    }
    
    newy <- predict.RWNN(object, ...)
    
    r <- newy - object$data$y
    if (tolower(type) %in% c("standard", "standardised", "rs")) {        
        r <- r / object$Sigma$Output
    }
    
    return(r)
}

#' @title Diagnostic-plots of an RWNN-object
#' 
#' @param x An \link{RWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments used by the function are '\code{X_val}' and '\code{y_val}', i.e. the features and targets of the validation-set. These are helpful when analysing whether overfitting of model has occured.  
#' 
#' @rdname plot.RWNN
#' @method plot RWNN
#'
#' @return NULL
#' 
#' @export
plot.RWNN <- function(x, ...) {
    dots <- list(...)
    if (is.null(dots$X_val) || is.null(dots$y_val)) {
        if (is.null(x$data)) {
            stop("The RWNN-object does not contain any data: Either supply 'X_val' and 'y_val', or re-create RWNN-object with 'include_data = TRUE' (default).")
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
