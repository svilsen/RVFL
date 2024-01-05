#########################################################################
####################### ERWNN neural networks AUX #######################
#########################################################################

#' @title Predicting targets of an ERWNN-object
#' 
#' @param object An \link{ERWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments '\code{newdata}', '\code{type}', and '\code{class}' can be specified as follows:
#' \describe{
#'   \item{\code{newdata}}{Expects a \link{matrix} or \link{data.frame} with the same features (columns) as in the original data.}
#'   \item{\code{type}}{A string taking the following values:
#'      \describe{
#'          \item{\code{"mean"}}{Returns the average prediction across all ensemble models.}
#'          \item{\code{"sd"}}{Returns the standard deviation of the predictions across all ensemble models.}
#'          \item{\code{"all"}}{Returns all predictions for each ensemble models.}
#'      }
#'   }
#'   \item{\code{class}}{A string taking the following values:
#'      \describe{
#'          \item{\code{"classify"}}{Returns the predicted class of ensemble. If used together with \code{type == "mean"}, the average prediction across the ensemble models are used to create the classification. However, if used with \code{type == "all"}, every ensemble is classified and returned.}
#'          \item{\code{"vote"}}{Returns the predicted class of ensemble by classifying each ensemble and using majority voting to make the final prediction.}
#'      }
#'   }
#' }
#' 
#' Furthermore, if '\code{class}' is set to either \code{"classify"} or \code{"vote"}, additional arguments '\code{t}' and '\code{b}' can be passed to the \link{classify}-function.
#' 
#' @return A matrix or a vector of predicted values depended on the arguments '\code{method}', '\code{type}', and '\code{class}'. 
#' 
#' @rdname predict.ERWNN
#' @method predict ERWNN
#' @export
predict.ERWNN <- function(object, ...) {
    #
    dots <- list(...)
    
    #
    type <- dots$type
    if (is.null(type)) {
        type <- "mean"
    } else {
        type <- tolower(type)
    }
    
    #
    if (is.null(dots$newdata)) {
        newdata <- object$data$X
    } else {
        if (is.null(object$formula)) {
            newdata <- as.matrix(dots$newdata)
        }
        else {
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
    o_type <- unique(do.call("c", sapply(object, function(x) x$Type)))
    if (object$method %in% c("bagging", "ed")) {
        if (type %in% c("a", "all", "f", "full")) {
            return(y_new)
        } else if (type %in% c("m", "mean", "avg", "average")) {
            W <- matrix(rep(object$weights, dim(newdata)[1]), ncol = B, byrow = TRUE)
            y_new <- matrix(apply(y_new * W, 1, sum), ncol = 1)
            
            if (o_type %in% c("c", "class", "classification")) {
                if ((!is.null(dots$type)) && (dots$type %in% c("c", "class", "classify"))) {
                    newy <- newy |> apply(1, which.max) |> (\(x) object$data$C[x])()
                }
            }
            
            return(y_new)
        } else if (type %in% c("s", "sd", "standarddeviation")) {
            y_new <- matrix(apply(y_new, 1, sd), ncol = 1)
        } else {
            stop("The value of 'type' was not valid, see '?predict.ERWNN' for valid options of 'type'.")
        }
    } else {
        W <- matrix(rep(object$weights, dim(newdata)[1]), ncol = B, byrow = TRUE)
        y_new <- matrix(apply(y_new * W, 1, sum), ncol = 1)
        
        if (o_type %in% c("c", "class", "classification")) {
            if ((!is.null(dots$type)) && (dots$type %in% c("c", "class", "classify"))) {
                newy <- newy |> apply(1, which.max) |> (\(x) object$data$C[x])()
            }
        }
        
        return(y_new)
    }
    
    return(y_new)
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
set_weights <- function(object, weights) {
    UseMethod("set_weights")
}

#' @rdname set_weights
#' @method set_weights ERWNN
#' 
#' @example inst/examples/sw_example.R
#'
#' @export
set_weights.ERWNN <- function(object, weights) {
    if (length(weights) != length(object$weights)) {
        stop("The length of 'weights' have to be equal to the number of ensemble weights.")
    }

    if (any(weights > 1) || any(weights < 0)) {
        stop("All weights have to be between 0 and 1.")
    }
    
    if (abs(sum(weights) - 1) > 1e-8) {
        stop("The weights have to sum to 1.")
    }
    
    object$weights <- weights
    return(object)
}
