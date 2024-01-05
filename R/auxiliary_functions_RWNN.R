########################################################################
####################### RWNN neural networks AUX #######################
########################################################################

#' @title Predicting targets of an RWNN-object
#' 
#' @param object An \link{RWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments used by the function are '\code{newdata}' and '\code{class}'. The argument '\code{newdata}' expects a \link{matrix} or \link{data.frame} with the same features (columns) as in the original data. While the '\code{class}' argument can be set to \code{"classify"}. If \code{class == "classify"} additional arguments '\code{t}' and '\code{b}' can be passed to the \link{classify}-function.
#' 
#' @return A vector of predicted values.
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
                newdata <- newdata[, keep, drop = FALSE]
            }
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
    if (object$Combined$Hidden) { 
        newH <- do.call("cbind", newH)
    } else {
        newH <- newH[[length(newH)]]
    }
    
    ## Estimate parameters in output layer
    if (object$Bias$Output) {
        newH <- cbind(1, newH)
    }
    
    newO <- newH
    if (object$Combined$Input) {
        newO <- cbind(newH, newdata)
    }
    
    newy <- newO %*% object$Weights$Output
    
    if (object$Type %in% c("c", "class", "classification")) {
        if (dots[["class"]] %in% c("c", "class", "classify")) {
            newp <- list(y = newy, C = object$data$C, t = dots[["t"]], b = dots[["b"]])
            newy <- do.call(classify, newp)
        }
    }
    
    return(newy)
}

#' @title Diagnostic-plots of an RWNN-object
#' 
#' @param x An \link{RWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The only additional argument used by the function is '\code{newdata}', which expects a matrix with the same number of features (columns) as in the original data.
#' 
#' @rdname plot.RWNN
#' @method plot RWNN
#'
#' @return NULL
#' 
#' @export
plot.RWNN <- function(x, ...) {
    dots <- list(...)
    if (is.null(dots$newdata)) {
        if (is.null(x$data)) {
            stop("The RWNN-object does not contain any data. Use the 'newdata' argument, or re-create 'RWNN-object' setting 'include_data = TRUE'.")
        }
        
        X_new <- x$data$X
        y_new <- x$data$y
        y_hat <- predict(x, newdata = X_new, classify = dots$classify)
    }
    else {
        newdata <- dots$newdata
        y_new <- newdata[, rownames(attr(terms(x$formula), "factors"))[attr(terms(x$formula), "response")]]
        y_hat <- predict(x, newdata = newdata, classify = dots$classify)
    }
    
    if (object$Type %in% c("c", "class", "classification")) {
        warning("The following figures are meant as diagnostic plots for models of the type 'regression', not the type 'classification'.")
    }
    
    if (is.null(dots$page)) {
        dev.hold()
        plot(y_hat ~ y_new, pch = 16, 
             xlab = "Observed targets", ylab = "Predicted targets")
        abline(0, 1, col = "black", lwd = 3)
        abline(0, 1, col = "dodgerblue", lwd = 2)
        dev.flush()
        
        readline(prompt = "Press [ENTER] for next plot...")
        dev.hold()
        plot(I(y_hat - y_new) ~ seq(length(y_new)), pch = 16,
             xlab = "Index", ylab = "Residual") 
        abline(0, 0, col = "black", lwd = 3)
        abline(0, 0, col = "dodgerblue", lwd = 2)
        dev.flush()
    }
    else if (dots$page == 1) {
        dev.hold()
        plot(y_hat ~ y_new, pch = 16, 
             xlab = "Observed targets", ylab = "Predicted targets")
        abline(0, 1, col = "black", lwd = 3)
        abline(0, 1, col = "dodgerblue", lwd = 2)
        dev.flush()
    }
    else if (dots$page == 2) {
        dev.hold()
        plot(I(y_hat - y_new) ~ seq(length(y_new)), pch = 16,
             xlab = "Index", ylab = "Residual") 
        abline(0, 0, col = "black", lwd = 3)
        abline(0, 0, col = "dodgerblue", lwd = 2)
        dev.flush()
    }
    else {
        stop("Invalid choice of 'page', it has to take the values 1, 2, or NULL.")
    }
    
    return(invisible(NULL))
}

