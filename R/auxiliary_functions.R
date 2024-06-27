###################################################
####################### AUX #######################
###################################################

#### ----
data_checks <- function(y, X) {
    if (!is.matrix(X)) {
        warning("'X' has to be a matrix... trying to cast 'X' as a matrix.")
        X <- as.matrix(X)
    }
    
    if (!is.matrix(y)) {
        warning("'y' has to be a matrix... trying to cast 'y' as a matrix.")
        y <- as.matrix(y)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    return(invisible(NULL))
}

#### ----
strip_terms <- function(formula) {
    attr_names <- names(attributes(formula))
    for (i in seq_along(attr_names)) {
        attr(formula, attr_names[i]) <- NULL
    }
    
    formula <- as.formula(formula)
    return(formula)
}

#### ----
orthonormal <- function(M) {
    # 
    svdM <- svd(M)
    U <- svdM$u
    S <- svdM$d
    
    #
    tol <- max(dim(M)) * max(S) * .Machine$double.eps
    R <- sum(S > tol)
    
    #
    X <- U[, 1:R, drop = FALSE]
    return(X)
}

random_orthonormal <- function(w, nr_rows, X, W_hidden, n_hidden, activation, bias_hidden) {
    W <- matrix(runif(n_hidden[w] * nr_rows), nrow = n_hidden[w])
    W <- orthonormal(W)
    
    if (nr_rows > n_hidden[w]) {
        if (w == 1) {
            Z <- X
        }
        else {
            Z <- rwnn_forward(X, W_hidden[seq_len(w - 1)], activation, bias_hidden)
            Z <- matrix(Z[[length(Z)]], ncol = n_hidden[w - 1])
        }
        
        if (bias_hidden[w]) {
            Z <- cbind(1, Z)
        }
        
        pca <- princomp(Z)
        L <- unname(t(pca$loadings[, seq_len(n_hidden[w]), drop = FALSE]))
        W <- W %*% L
    }
    
    W <- t(W)
    return(W)
}

#### ---- 
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
    
    if (any(is.null(dots$newdata))) {
        if (is.null(object$data)) {
            stop("The RWNN-object does not contain any data. Use the 'newdata' argument, or re-create 'RWNN-object' setting 'include_data = TRUE' (default).")
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
        
        if (dim(newdata)[2] != (dim(object$weights$W[[1]])[1] - as.numeric(object$bias$W[1]))) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
    }
    
    newH <- rwnn_forward(
        X = newdata, 
        W = object$weights$W, 
        activation = object$activation,
        bias = object$bias$W
    )
    
    newH <- lapply(seq_along(newH), function(i) matrix(newH[[i]], ncol = object$n_hidden[i]))
    if (object$combined$W) { 
        newH <- do.call("cbind", newH)
    } else {
        newH <- newH[[length(newH)]]
    }
    
    ## Estimate parameters in output layer
    if (object$bias$beta) {
        newH <- cbind(1, newH)
    }
    
    newO <- newH
    if (object$combined$X) {
        newO <- cbind(newH, newdata)
    }
    
    newy <- newO %*% object$weights$beta
    
    if (object$type %in% c("c", "class", "classification")) {
        if (!is.null(dots[["class"]])) {
            if (dots[["class"]] %in% c("c", "class", "classify")) {
                newp <- list(y = newy, C = object$data$C, t = dots[["t"]], b = dots[["b"]])
                newy <- do.call(classify, newp)
            }
        }
    }
    
    return(newy)
}

#### ----
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
#'          \item{\code{"std"}}{Returns the standard deviation of the predictions across all ensemble models.}
#'          \item{\code{"all"}}{Returns all predictions for each ensemble models.}
#'      }
#'   }
#'   \item{\code{class}}{A string taking the following values:
#'      \describe{
#'          \item{\code{"classify"}}{Returns the predicted class of ensemble. If used together with \code{type = "mean"}, the average prediction across the ensemble models are used to create the classification. However, if used with \code{type = "all"}, every ensemble is classified and returned.}
#'          \item{\code{"voting"}}{Returns the predicted class of ensemble by classifying each ensemble and using majority voting to make the final prediction, i.e. the \code{type} argument is overruled.}
#'      }
#'   }
#' }
#' 
#' Furthermore, if '\code{class}' is set to either \code{"classify"} or \code{"voting"}, additional arguments '\code{t}' and '\code{b}' can be passed to the \link{classify}-function.
#' 
#' NB: if the ensemble is created using the \link{boost_rwnn}-function, then \code{type} should be set to \code{"mean"}.
#' 
#' @return An list, matrix, or vector of predicted values depended on the arguments '\code{method}', '\code{type}', and '\code{class}'. 
#' 
#' @rdname predict.ERWNN
#' @method predict ERWNN
#' @export
predict.ERWNN <- function(object, ...) {
    #
    dots <- list(...)
    
    #
    type <- dots[["type"]]
    if (is.null(type)) {
        type <- "mean"
    } else if (!is.null(dots[["class"]])) {
        if (dots[["class"]] %in% c("v", "vote", "voting")) {
            type <- "all"
        }
    } else {
        type <- tolower(type)
    }
    
    #
    if (is.null(dots[["newdata"]])) {
        newdata <- object$data$X
    } else {
        if (is.null(object$formula)) {
            newdata <- as.matrix(dots$newdata)
        } else {
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
            
            newdata <- as.matrix(newdata, ncol = length(keep))
        }
        
        if (dim(newdata)[2] != (dim(object$models[[1]]$weights$W[[1]])[1] - as.numeric(object$models[[1]]$bias$W[1]))) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
    }
    
    ## Set-up
    o_type <- unique(sapply(object$models, function(x) x$type))
    if (length(o_type) > 1) {
        o_type <- o_type[1]
        warning("Multiple 'type' fields found among the ensemble models; therefore, only the first ensemble model is used to determine model type.")
    }
    
    B <- length(object$weights)
    
    ## Prediction based on type and class.
    if (type %in% c("a", "all")) {
        y_new <- vector("list", B)
        for (b in seq_len(B)) {
            y_new_b <- predict.RWNN(object = object$models[[b]], newdata = newdata)
            
            if (o_type %in% c("c", "class", "classification")) {
                if (!is.null(dots[["class"]])) {
                    if (dots[["class"]] %in% c("c", "class", "classify", "v", "vote", "voting")) {
                        p_new_b <- list(y = y_new_b, C = object$data$C, t = dots[["t"]], b = dots[["b"]])
                        y_new_b <- do.call(classify, p_new_b)
                    }
                }
            }
            
            y_new[[b]] <- y_new_b
        }
        
        if (o_type %in% c("c", "class", "classification")) {
            if (!is.null(dots[["class"]])) {
                if (dots[["class"]] %in% c("v", "vote", "voting")) {
                    y_new <- do.call("cbind", y_new)
                    y_new <- apply(y_new, 1, mode)
                }
            }
        }
        
        return(y_new)
    }
    else if (type %in% c("m", "mean")) {
        y_new <- matrix(0, nrow = dim(newdata)[1], ncol = dim(object$data$y)[2])
        for (b in seq_len(B)) {
            y_new_b <- predict.RWNN(object = object$models[[b]], newdata = newdata)
            y_new <- y_new + object$weights[b] * y_new_b
        }
        
        if (o_type %in% c("c", "class", "classification")) {
            if (!is.null(dots[["class"]])) {
                if (dots[["class"]] %in% c("c", "class", "classify")) {
                    p_new <- list(y = y_new, C = object$data$C, t = dots[["t"]], b = dots[["b"]])
                    y_new <- do.call(classify, p_new)
                }
            }
        }
        
        return(y_new)
    }
    else if (type %in% c("s", "std", "standarddeviation")) {
        y_new <- matrix(0, nrow = dim(newdata)[1], ncol = dim(object$data$y)[2])
        y_sq_new <- matrix(0, nrow = dim(newdata)[1], ncol = dim(object$data$y)[2])
        for (b in seq_len(B)) {
            y_new_b <- predict.RWNN(object = object$models[[b]], newdata = newdata)
            
            y_new <- y_new + object$weights[b] * y_new_b
            y_sq_new <- y_sq_new + object$weights[b] * y_new_b^2
        }
        
        N <- sum(abs(object$weights) > 1e-8)
        W <- (N - 1) * sum(object$weights) / N
        
        s_new <- (y_sq_new - y_new^2) / W
        return(s_new)
    }
    else {
        stop("The value of 'type' was not valid, see '?predict.ERWNN' for valid options of 'type'.")
    }
}

#### ----
#' Classifier
#' 
#' @description Function classifying an observation.
#' 
#' @param y A matrix of predicted classes.
#' @param C A vector of class names corresponding to the columns of \code{y}.
#' @param t The decision threshold which the predictions have to exceed (default is '0'). 
#' @param b A buffer which the largest prediction has to exceed when compared to the second largest prediction (default is '0').
#' 
#' @return A vector of class predictions.
#' 
#' @export 
classify <- function(y, C, t = NULL, b = NULL) {
    #
    if (dim(y)[2] != length(C)) {
        stop("The number of columns 'y' has to match the number of elements in 'C'.")
    }
    
    # 
    if (is.null(t)) {
        t <- 0.0
    }
    
    #
    if (is.null(b)) {
        b <- 0.0
    }
    
    #
    yc <- classify_cpp(y, C, t, b)
    return(yc)
}

#### ----
#' @title Set ensemble weights of an ERWNN-object
#' 
#' @description Manually set ensemble weights of an \link{ERWNN-object}.
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

#### ----
#' @title Estimate ensemble weights of an ERWNN-object
#' 
#' @description Estimate ensemble weights of an \link{ERWNN-object}.
#' 
#' @param object An \link{ERWNN-object}.
#' @param data_val A data.frame or tibble containing the validation-set.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
estimate_weights <- function(object, data_val = NULL) {
    UseMethod("estimate_weights")
}

#' @rdname estimate_weights
#' @method estimate_weights ERWNN
#' 
#' @example inst/examples/ew_example.R
#'
#' @export
estimate_weights.ERWNN <- function(object, data_val = NULL) {
    if (is.null(data_val)) {
        warning("The validation-set was not properly specified, therefore, the training-set is used for weight estimation.")
        
        X_val <- object$data$X
        y_val <- object$data$y
    }
    else {
        X_val <- data_val[, all.vars(mm$formula)[-1], drop = FALSE]
        y_val <- data_val[, all.vars(mm$formula)[1], drop = FALSE]
    }
    
    B <- length(object$models)
    C <- predict(object, newdata = X_val, type = "all")
    C <- do.call("cbind", C)

    y <- y_val |> as.matrix()
    
    object$weights <- estimate_weights_stack(C = C, b = y, B = B)
    return(object)
}
