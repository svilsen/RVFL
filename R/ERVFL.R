############################################################################
####################### Ensemble RVFL neural network #######################
############################################################################

#### Bagging ----

#' @title Bagging random vector functional links.
#' 
#' @description Use bootstrap aggregation to reduce the variance of random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of bootstrap samples.
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param N_features The number of features randomly chosen in each iteration (default is \code{ceiling(ncol(X) / 3)}). 
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An ERVFL-object containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list of \link{RVFL}-objects.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method ('bagging' in this case)}
#' }
#' 
#' @export
bagRVFL <- function(X, y, N_hidden, B = 100, lambda = 0, N_features = NULL, ...) {
    UseMethod("bagRVFL")
}

#' @rdname bagRVFL
#' @method bagRVFL default
#' 
#' @example inst/examples/bagrvfl_example.R
#' 
#' @export
bagRVFL.default <- function(X, y, N_hidden, B = 100, lambda = 0, N_features = NULL, ...) {
    ## Checks
    # Data
    if (!is.matrix(X)) {
        warning("'X' has to be a matrix... trying to cast 'X' as a matrix.")
        X <- as.matrix(X)
    }
    
    if (!is.matrix(y)) {
        warning("'y' has to be a matrix... trying to cast 'y' as a matrix.")
        y <- as.matrix(y)
    }
    
    if (dim(y)[2] != 1) {
        warning("More than a single column was detected in 'y', only the first column is used in the model.")
        y <- matrix(y[, 1], ncol = 1)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    if (is.null(B)) {
        B <- 100
        
        warning(paste0("Note: 'B' was not supplied, 'B' was set to ", B, "."))
    }
    
    if (is.null(N_features)) {
        N_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        indices_b <- sample(nrow(X), nrow(X), replace = TRUE)
        
        X_b <- matrix(X[indices_b, ], ncol = ncol(X))
        y_b <- matrix(y[indices_b], ncol = ncol(y))    
        
        objects[[b]] <- RVFL(X = X_b, y = y_b, N_hidden = N_hidden, lambda = lambda, N_features = N_features, ...)
    }
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = objects, 
        weights = rep(1L / B, B), 
        method = "bagging"
    )  
    
    class(object) <- "ERVFL"
    return(object)
}

#### Boosting ----

#' @title Boosting random vector functional links
#' 
#' @description Use gradient boosting to create ensemble random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of levels used in the boosting tree.
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param epsilon The learning rate.
#' @param N_features The number of features randomly chosen in each iteration (default is \code{ceiling(ncol(X) / 3)}).
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An ERVFL-object containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list of \link{RVFL}-objects.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method ('boosting' in this case)}
#' }
#' 
#' @export
boostRVFL <- function(X, y, N_hidden, B = 10, lambda = 0, epsilon = 1, N_features = NULL, ...) {
    UseMethod("boostRVFL")
}

#' @rdname boostRVFL
#' @method boostRVFL default
#' 
#' @example inst/examples/boostrvfl_example.R
#' 
#' @export
boostRVFL.default <- function(X, y, N_hidden, B = 10, lambda = 0, epsilon = 1, N_features = NULL, ...) {
    ## Checks
    # Data
    if (!is.matrix(X)) {
        warning("'X' has to be a matrix... trying to cast 'X' as a matrix.")
        X <- as.matrix(X)
    }
    
    if (!is.matrix(y)) {
        warning("'y' has to be a matrix... trying to cast 'y' as a matrix.")
        y <- as.matrix(y)
    }
    
    if (dim(y)[2] != 1) {
        warning("More than a single column was detected in 'y', only the first column is used in the model.")
        y <- matrix(y[, 1], ncol = 1)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    if (is.null(B)) {
        B <- 10
        
        warning(paste0("Note: 'B' was not supplied, 'B' was set to ", B, "."))
    }
    
    if (is.null(epsilon)) {
        epsilon <- 1
        warning("Note: 'epsilon' was not supplied and set to 1.")
    }
    else if (epsilon > 1) {
        epsilon <- 1
        warning("'epsilon' has to be a number between 0 and 1.")
    }
    else if (epsilon < 0) {
        epsilon <- 0
        warning("'epsilon' has to be a number between 0 and 1.")
    }
    
    if (is.null(N_features)) {
        N_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        X_b <- X
        if (b == 1) {
            y_b <- y
        }
        else {
            y_b <- y_b - epsilon * predict(objects[[b - 1]])
        }
        
        objects[[b]] <- RVFL(X = X_b, y = y_b, N_hidden = N_hidden, lambda = lambda, N_features = N_features, ...)
    }
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = objects, 
        weights = rep(1L / B, B), 
        method = "boosting"
    )  
    
    class(object) <- "ERVFL"
    return(object)
}


#### Stacking ----
#' @title Stacking random vector functional links
#' 
#' @description Use stacking to create ensemble random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of models in the stack.
#' @param folds The number of folds used to train the RVFL models. 
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param N_features The number of features randomly chosen in each iteration (default is \code{ceiling(ncol(X) / 3)}).
#' @param optimise TRUE/FALSE: Should the stacking weights be optimised (or should the stack just use the average)? 
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An ERVFL-object containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list of \link{RVFL}-objects.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method ('boosting' in this case)}
#' }
#' 
#' @export
stackRVFL <- function(X, y, N_hidden, B = 100, folds = 10, lambda = 0, N_features = NULL, optimise = TRUE, ...) {
    UseMethod("stackRVFL")
}

#' @rdname stackRVFL
#' @method stackRVFL default
#' 
#' @example inst/examples/stackrvfl_example.R
#' 
#' @export
stackRVFL.default <- function(X, y, N_hidden, B = 100, folds = 10, lambda = 0, N_features = NULL, optimise = TRUE, ...) {
    ## Checks
    # Data
    if (!is.matrix(X)) {
        warning("'X' has to be a matrix... trying to cast 'X' as a matrix.")
        X <- as.matrix(X)
    }
    
    if (!is.matrix(y)) {
        warning("'y' has to be a matrix... trying to cast 'y' as a matrix.")
        y <- as.matrix(y)
    }
    
    if (dim(y)[2] != 1) {
        warning("More than a single column was detected in 'y', only the first column is used in the model.")
        y <- matrix(y[, 1], ncol = 1)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    if (is.null(B)) {
        B <- 100
        
        warning(paste0("Note: 'B' was not supplied, 'B' was set to ", B, "."))
    }
    
    if (is.null(N_features)) {
        N_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    fold_index <- create_folds(X, folds)
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        object_b <- RVFL(X = X, y = y, N_hidden = N_hidden, lambda = lambda, N_features = N_features, ...)
        
        beta_b <- vector("list", folds)
        for (k in seq_len(folds)) {
            Xk <- matrix(X[-fold_index[[k]], ], ncol = ncol(X))
            yk <- matrix(y[-fold_index[[k]], ], ncol = ncol(y))
            
            Hk <- rvfl_forward(Xk, object_b$Weights$Hidden, object_b$activation, object_b$Bias$Hidden)
            Hk <- lapply(seq_along(Hk), function(i) matrix(Hk[[i]], ncol = N_hidden[i]))
            Hk <- do.call("cbind", Hk)
            
            ## Estimate parameters in output layer
            if (object_b$Bias$Output) {
                Hk <- cbind(1, Hk)
            }
            
            Ok <- Hk
            if (object_b$Combined) {
                Ok <- cbind(Xk, Hk)
            }
            
            beta_b[[k]] <- estimate_output_weights(Ok, yk, lambda)$beta
        }
        
        object_b$Weights$Output <- apply(do.call("cbind", beta_b), 1, mean)
        objects[[b]] <- object_b
    }
    
    ##
    if (optimise) {
        C <- do.call("cbind", lapply(objects, predict))
        w <- estimate_weights_stack(C = C, b = y, B = B)
    }
    else {
        w <- 1 / B
    }
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = objects, 
        weights = w, 
        method = "stacking"
    )  
    
    class(object) <- "ERVFL"
    return(object)
}

#### ED ----
#' @title Ensemble deep random vector functional links
#' 
#' @description Use multiple layers to create deep ensemble random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An ERVFL-object containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list of \link{RVFL}-objects.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method ('boosting' in this case)}
#' }
#' 
#' @export
edRVFL <- function(X, y, N_hidden, lambda = 0, ...) {
    UseMethod("edRVFL")
}

#' @rdname edRVFL
#' @method edRVFL default
#' 
#' @example inst/examples/edrvfl_example.R
#' 
#' @export
edRVFL.default <- function(X, y, N_hidden, lambda = 0, ...) {
    ## Checks
    # Data
    if (!is.matrix(X)) {
        warning("'X' has to be a matrix... trying to cast 'X' as a matrix.")
        X <- as.matrix(X)
    }
    
    if (!is.matrix(y)) {
        warning("'y' has to be a matrix... trying to cast 'y' as a matrix.")
        y <- as.matrix(y)
    }
    
    if (dim(y)[2] != 1) {
        warning("More than a single column was detected in 'y', only the first column is used in the model.")
        y <- matrix(y[, 1], ncol = 1)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    ##
    deepRVFL <- RVFL(X = X, y = y, N_hidden = N_hidden, lambda = lambda, ...)
    H <- rvfl_forward(X = X, W = deepRVFL$Weights$Hidden, activation = deepRVFL$activation, bias = deepRVFL$Bias$Hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = deepRVFL$N_hidden[i]))
    O <- lapply(seq_along(H), function(i) cbind(1, X, H[[i]]))
    beta <- lapply(seq_along(O), function(i) estimate_output_weights(O[[i]], y, lambda)$beta)
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RVFLmodels = deepRVFL, 
        OutputWeights = beta,
        weights = rep(1L / length(N_hidden), length(N_hidden)), 
        method = "ed"
    ) 
    
    class(object) <- "ERVFL"
    return(object)
}


#### Auxiliary ----

create_folds <- function(X, folds) {
    index <- sample(nrow(X), nrow(X), replace = FALSE)
    fold_index <- rep(seq_len(folds), each = floor(nrow(X) / folds))
    
    if (length(fold_index) < length(index)) {
        fold_index <- c(fold_index, seq_len(folds)[seq_len(length(index) - length(fold_index))])
    }
    
    return(unname(split(x = index, f = fold_index)))
}

#' @title Coefficients of the ERVFL object.
#' 
#' @param object An ERVFL-object.
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
#' If '\code{method}' was \code{"boosting"}, a matrix is returned corresponding to '\code{type == "all"}'.
#' 
#' @rdname coef.ERVFL
#' @method coef ERVFL
#' @export
coef.ERVFL <- function(object, ...) {
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
        B <- length(object$RVFLmodels)
        beta <- vector("list", B)
        for (b in seq_along(beta)) {
            beta[[b]] <- coef(object$RVFLmodels[[b]])
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
            stop("The passed value of 'type' was not valid. See '?coef.ERVFL' for valid options of 'type'.")
        }
    }
    else {
        return(beta)
    }
}

#' @title Predicting targets of an ERVFL object.
#' 
#' @param object An ERVFL-object.
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
#' If '\code{method}' was \code{"bagging"}, the '\code{type}' yields the following results: 
#' \describe{
#'     \item{\code{"mean" (default):}}{A vector containing the weighted (using the \code{weights} element of the ERVFL-object) sum each observation taken across the bootstrap samples.}
#'     \item{\code{"sd":}}{A vector containing the standard deviation of each prediction taken across the bootstrap samples.}
#'     \item{\code{"all":}}{A matrix where every column contains the predicted values corresponding to each of the boostrapped models.}
#' }
#' 
#' If '\code{method}' was \code{"boosting"}, a vector is returned each element being the sum of the boosted predictions.
#'
#' @rdname predict.ERVFL
#' @method predict ERVFL
#' @export
predict.ERVFL <- function(object, ...) {
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
        if (dim(dots$newdata)[2] != dim(object$data$X)[2]) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
        
        newdata <- dots$newdata 
        
        if (!is.matrix(newdata)) {
            newdata <- as.matrix(newdata)
        }
    }
    
    ##
    B <- length(object$weights)
    if (object$method == "ed") {
        newH <- rvfl_forward(X = newdata, W = object$RVFLmodels$Weights$Hidden, activation = object$RVFLmodels$activation, bias = object$RVFLmodels$Bias$Hidden)
        newH <- lapply(seq_along(newH), function(i) matrix(newH[[i]], ncol = object$RVFLmodels$N_hidden[i]))
        newO <- lapply(seq_along(newH), function(i) cbind(1, newdata, newH[[i]]))
        y_new <- lapply(seq_along(newO), function(i) newO[[i]] %*% object$OutputWeights[[i]])
        
        y_new <- do.call("cbind", y_new)
    }
    else {
        y_new <- vector("list", B)
        for (b in seq_along(y_new)) {
            y_new[[b]] <- predict.RVFL(object = object$RVFLmodels[[b]], newdata = newdata)
        }
        
        y_new <- do.call("cbind", y_new)
    }
    
    ##
    if (object$method %in% c("bagging", "stacking", "ed", "resample")) {
        if (type %in% c("a", "all", "f", "full")) {
            return(y_new)
        }
        else if (type %in% c("m", "mean", "avg", "average")) {
            W <- matrix(rep(object$weights, dim(newdata)[1]), ncol = B, byrow = TRUE)
            y_new <- matrix(apply(y_new * W, 1, sum), ncol = 1)
            return(y_new)
        }
        else if (type %in% c("s", "sd", "standarddeviation")) {
            y_new <- matrix(apply(y_new, 1, sd), ncol = 1)
            return(y_new)
        }
        else {
            stop("The passed value of 'type' was not valid. See '?coef.ERVFL' for valid options of 'type'.")
        }
    }
    else {
        y_new <- matrix(apply(y_new, 1, sum), ncol = 1)
        return(y_new)
    }
}

#' @title Residuals of the ERVFL object.
#' 
#' @param object An ERVFL-object.
#' @param ... Additional arguments.
#' 
#' @details No additional arguments are used in this instance.
#' 
#' @return A vector of raw residuals between the predicted (using \code{type = "mean"}) and observed targets.
#' 
#' @rdname residuals.ERVFL
#' @method residuals ERVFL
#' @export
residuals.ERVFL <- function(object, ...) {
    dots <- list(...)
    y_new <- predict(object)
    
    r <- y_new - object$data$y
    return(r)
}

#' @title Diagnostic-plots of an ERVFL-object.
#' 
#' @param x An ERVFL-object.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments used by the function are '\code{X_val}' and '\code{y_val}', i.e. the features and targets of the validation-set. These are helpful when analysing whether overfitting of model has occurred.  
#' 
#' @return NULL
#' 
#' @rdname plot.ERVFL
#' @method plot ERVFL
#'
#' @export
plot.ERVFL <- function(x, ...) {
    dots <- list(...)
    if (is.null(dots$X_val) && is.null(dots$y_val)) {
        X_val <- x$data$X
        y_val <- x$data$y
    }
    else if (is.null(dots$X_val) || is.null(dots$y_val)) {
        message("The testing-set was not properly specified, therefore, the training-set is used.")
        
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
        
        readline(prompt = "Press [ENTER] for next plot...")
        dev.hold()
        plot(x$weights ~ seq(length(x$weights)), pch = 16,
             xlab = "Bootstrap index", ylab = "Weights") 
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
    else if (dots$page == 3) {
        dev.hold()
        plot(x$weights ~ seq(length(x$weights)), pch = 16,
             xlab = "Bootstrap index", ylab = "Weights") 
        dev.flush()
    }
    else {
        stop("Invalid choice of 'page', it has to take the values 1, 2, 3, or NULL.")
    }
    
    return(invisible(NULL))
}


