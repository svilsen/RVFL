############################################################################
####################### Bagging ERWNN neural network #######################
############################################################################

#' @title Bagging random weight neural networks
#' 
#' @description Use bootstrap aggregation to reduce the variance of random weight neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of each RWNN.
#' @param B The number of bootstrap samples.
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
bag_rwnn <- function(X, y, formula, data, N_hidden = c(), lambda = NULL, B = 100, control = list()) {
    UseMethod("bag_rwnn")
}

#' @rdname bag_rwnn
#' @method bag_rwnn default
#' 
#' @example inst/examples/bagrwnn_example.R
#' 
#' @export
bag_rwnn.default <- function(X, y, N_hidden = c(), lambda = NULL, B = 100, control = list()) {
    ## Checks
    dc <- data_checks(y, X)
    
    if (is.null(B) | !is.numeric(B)) {
        B <- 100
        warning("Note: 'B' was not supplied, 'B' was set to 100.")
    }
    
    if (is.null(control$N_features)) {
        control$N_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        indices_b <- sample(nrow(X), nrow(X), replace = TRUE)
        
        X_b <- matrix(X[indices_b, ], ncol = ncol(X))
        y_b <- matrix(y[indices_b], ncol = ncol(y))    
        
        rwnn_b <- rwnn(X = X_b, y = y_b, N_hidden = N_hidden, lambda = lambda, control = control)
        objects[[b]] <- rwnn_b
    }
    
    ##
    object <- list(
        formula = NULL,
        data = list(X = X, y = y), 
        RWNNmodels = objects, 
        weights = rep(1L / B, B), 
        method = "bagging"
    )  
    
    class(object) <- "ERWNN"
    return(object)
}


#' @rdname bag_rwnn
#' @method bag_rwnn formula
#' 
#' @example inst/examples/bagrwnn_example.R
#' 
#' @export
bag_rwnn.formula <- function(formula, data, N_hidden = c(), lambda = NULL, B = 100, control = list()) {
    if (missing(formula)) {
        stop("'formula' needs to be supplied when using 'data'.")
    }
    
    if (missing(data)) {
        stop("'data' needs to be supplied when using 'formula'.")
    }
    
    # Re-capture feature names when '.' is used in formula interface
    formula <- terms(formula, data = data)
    formula <- strip_terms(formula)
    
    #
    X <- model.matrix(formula, data)
    keep <- which(colnames(X) != "(Intercept)")
    if (any(colnames(X) == "(Intercept)")) {
        X <- X[, keep]
    }
    
    X <- as.matrix(X, ncol = length(keep))
    
    #
    y <- as.matrix(model.response(model.frame(formula, data)), nrow = nrow(data))
    
    #
    mm <- bag_rwnn(X, y, N_hidden = N_hidden, lambda = lambda, B = B, control = control)
    mm$formula <- formula
    return(mm)
}