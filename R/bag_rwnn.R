############################################################################
####################### Bagging ERWNN neural network #######################
############################################################################

#' @title Bagging random weight neural networks
#' 
#' @description Use bootstrap aggregation to reduce the variance of random weight neural network models.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param n_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of each RWNN.
#' @param B The number of bootstrap samples.
#' @param method The penalisation type passed to \link{ae_rwnn}. Set to \code{NULL} (default), \code{"l1"}, or \code{"l2"}. If \code{NULL}, the \link{rwnn} is used as the base learner.
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
bag_rwnn <- function(formula, data = NULL, n_hidden = c(), lambda = NULL, B = 100, method = NULL, type = NULL, control = list()) {
    UseMethod("bag_rwnn")
}

bag_rwnn_matrix <- function(X, y, n_hidden = c(), lambda = NULL, B = 100, method = NULL, type = NULL, control = list()) {
    ## Checks
    if (is.null(control[["include_data"]])) {
        control$include_data <- FALSE
    }
    
    dc <- data_checks(y, X)
    
    if (is.null(B) | !is.numeric(B)) {
        B <- 100
        warning("Note: 'B' was not supplied and is therefore set to 100.")
    }
    
    if (is.null(control$n_features)) {
        control$n_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    N <- nrow(X)
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        indices_b <- sample(N, N, replace = TRUE)
        
        X_b <- X[indices_b, , drop = FALSE]
        y_b <- y[indices_b, , drop = FALSE]  
        
        if (is.null(method)) {
            rwnn_b <- rwnn_matrix(X = X_b, y = y_b, n_hidden = n_hidden, lambda = lambda, type = type, control = control)
        }
        else {
            rwnn_b <- ae_rwnn_matrix(X = X_b, y = y_b, n_hidden = n_hidden, lambda = lambda, method = method, type = type, control = control)
        }
        
        objects[[b]] <- rwnn_b
    }
    
    ##
    object <- list(
        formula = NULL,
        data = list(X = X, y = y, C = ifelse(type == "regression", NA, colnames(y))), 
        models = objects, 
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
bag_rwnn.formula <- function(formula, data = NULL, n_hidden = c(), lambda = NULL, B = 100, method = NULL, type = NULL, control = list()) {
    # Checks for 'n_hidden'
    if (length(n_hidden) < 1) {
        stop("When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see ?lm.")
    }
    
    if (!is.numeric(n_hidden)) {
        stop("Not all elements of the 'n_hidden' vector were numeric.")
    }
    
    # Checks for 'data'
    if (is.null(data)) {
        data <- tryCatch(
            expr = {
                as.data.frame(as.matrix(model.frame(formula)))
            },
            error = function(e) {
                message("'data' needs to be supplied when using 'formula'.")
            }
        )
        
        x_name <- paste0(attr(terms(formula), "term.labels"), ".")
        colnames(data) <- paste0("V", gsub(x_name, "", colnames(data)))
        colnames(data)[1] <- "y"
        
        formula <- paste(colnames(data)[1], "~", paste(colnames(data)[seq_along(colnames(data))[-1]], collapse = " + "))
        formula <- as.formula(formula)
        warning("'data' was supplied through the formula interface, not a 'data.frame', therefore, the columns of the feature matrix and the response have been renamed.")
    }
    
    # Checks for 'method'
    if (!is.null(method)) {
        method <- tolower(method)
        if (!(method %in% c("l1", "l2"))) {
            stop("'method' has to be set to 'NULL', 'l1', or 'l2'.")
        }
    }
    
    # Re-capture feature names when '.' is used in formula interface
    formula <- terms(formula, data = data)
    formula <- strip_terms(formula)
    
    #
    X <- model.matrix(formula, data)
    keep <- which(colnames(X) != "(Intercept)")
    if (any(colnames(X) == "(Intercept)")) {
        X <- X[, keep, drop = FALSE]
    }
    
    #
    y <- model.response(model.frame(formula, data))
    y <- as.matrix(y, nrow = nrow(data))
    
    #
    if (is.null(type)) {
        if (is(y[, 1], "numeric")) {
            type <- "regression"
            
            if (all(abs(y - round(y)) < 1e-8)) {
                warning("The response consists of only integers, is this a classification problem?")
            }
        }
        else if (class(y[, 1]) %in% c("factor", "character", "logical")) {
            type <- "classification"
        }
    }
    
    # Change output based on 'type'
    if (tolower(type) %in% c("c", "class", "classification")) {
        type <- "classification"
        
        y_names <- sort(unique(y))
        y <- factor(y, levels = y_names)
        y <- model.matrix(~ 0 + y)
        
        attr(y, "assign") <- NULL
        attr(y, "contrasts") <- NULL
        
        y <- 2 * y - 1
        
        colnames(y) <- paste(y_names, sep = "")
    } 
    else if (tolower(type) %in% c("r", "reg", "regression")) {
        type <- "regression"
    }
    else {
        stop("'type' has not been correctly specified, it needs to be set to either 'regression' or 'classification'.")
    }
    
    #
    mm <- bag_rwnn_matrix(X, y, n_hidden = n_hidden, lambda = lambda, B = B, method = method, type = type, control = control)
    mm$formula <- formula
    return(mm)
}