#############################################################################
####################### Boosting ERWNN neural network #######################
#############################################################################

#' @title Boosting random weight neural networks
#' 
#' @description Use gradient boosting to create ensemble random weight neural network models.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of each RWNN.
#' @param B The number of levels used in the boosting tree.
#' @param epsilon The learning rate.
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
boost_rwnn <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, B = 10, epsilon = 1, control = list()) {
    UseMethod("boost_rwnn")
}

boost_rwnn.matrix <- function(X, y, N_hidden = c(), lambda = NULL, B = 10, epsilon = 1, control = list()) {
    ## Checks
    dc <- data_checks(y, X)
    
    if (is.null(B) | !is.numeric(B)) {
        B <- 10
        warning("Note: 'B' was not supplied, 'B' was set to 10.")
    }
    
    if (is.null(epsilon) | !is.numeric(epsilon)) {
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
    
    if (is.null(control$N_features)) {
        control$N_features <- ceiling(dim(X)[2] / 3)
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
        
        objects[[b]] <- rwnn.matrix(X = X_b, y = y_b, N_hidden = N_hidden, lambda = lambda, control = control)
    }
    
    ##
    object <- list(
        formula = NULL,
        data = list(X = X, y = y), 
        RWNNmodels = objects, 
        weights = rep(1L / B, B), 
        method = "boosting"
    )  
    
    class(object) <- "ERWNN"
    return(object)
}

#' @rdname boost_rwnn
#' @method boost_rwnn formula
#' 
#' @example inst/examples/boostrwnn_example.R
#' 
#' @export
boost_rwnn.formula <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, B = 10, epsilon = 1, control = list()) {
    if (is.null(data)) {
        data <- tryCatch(
            expr = {
                model.matrix(formula)
            },
            error = function(e) {
                message("'data' needs to be supplied when using 'formula'.")
            }
        )
        
        data <- as.data.frame(data)
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
    mm <- boost_rwnn.matrix(X, y, N_hidden = N_hidden, lambda = lambda, B = B, epsilon = epsilon, control = control)
    mm$formula <- formula
    return(mm)
}