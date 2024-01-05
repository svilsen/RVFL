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
#' @param method The penalisation type passed to \link{ae_rwnn}. Set to \code{NULL} (default), \code{"l1"}, or \code{"l2"}. If \code{NULL}, the \link{rwnn} is used as the base learner.
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
boost_rwnn <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, B = 10, epsilon = 1, method = NULL, type = NULL, control = list()) {
    UseMethod("boost_rwnn")
}

boost_rwnn.matrix <- function(X, y, N_hidden = c(), lambda = NULL, B = 10, epsilon = 1, method = NULL, type = NULL, control = list()) {
    ## Checks
    dc <- data_checks(y, X)
    
    if (is.null(B) | !is.numeric(B)) {
        B <- 10
        warning("Note: 'B' was set to '10', as it was not supplied.")
    }
    
    if (is.null(epsilon) | !is.numeric(epsilon)) {
        epsilon <- 1
        warning("Note: 'epsilon' was set to '1', as it was not supplied.")
    }
    else if (epsilon > 1) {
        epsilon <- 1
        warning("'epsilon' has to be a number between '0' and '1'.")
    }
    else if (epsilon < 0) {
        epsilon <- 0
        warning("'epsilon' has to be a number between '0' and '1'.")
    }
    
    if (is.null(control$N_features)) {
        control$N_features <- dim(X)[2] 
    }
    
    ##
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        if (b == 1) {
            y_b <- y
        } else {
            y_b <- y_b - epsilon * predict(objects[[b - 1]])
        }
        
        if (is.null(method)) {
            objects[[b]] <- rwnn.matrix(X = X, y = y_b, N_hidden = N_hidden, lambda = lambda, type = type, control = control)
        }
        else {
            objects[[b]] <- ae_rwnn.matrix(X = X, y = y_b, N_hidden = N_hidden, lambda = lambda, method = method, type = type, control = control)
        }
        
    }
    
    ##
    object <- list(
        formula = NULL,
        data = list(X = X, y = y, C = colnames(y)), 
        RWNNmodels = objects, 
        weights = c(rep(epsilon, B - 1), 1L), 
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
boost_rwnn.formula <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, B = 10, epsilon = 0.1, method = NULL, type = NULL, control = list()) {
    # Checks for 'N_hidden'
    if (length(N_hidden) < 1) {
        stop("When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see ?lm.")
    }
    
    if (!is.numeric(N_hidden)) {
        stop("Not all elements of the 'N_hidden' vector were numeric.")
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
    if (is.null(type)) {
        if (class(y) == "numeric") {
            type <- "regression"
            
            if (all(abs(y - round(y)) < 1e-8)) {
                warning("The response consists of only integers, is this a classification problem?")
            }
        }
        else if (class(y) %in% c("factor", "character", "logical")) {
            type <- "classification"
        }
    }
    
    y <- as.matrix(y, nrow = nrow(data))
    
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
    mm <- boost_rwnn.matrix(X, y, N_hidden = N_hidden, lambda = lambda, B = B, epsilon = epsilon, method = method, type = type, control = control)
    mm$formula <- formula
    return(mm)
}