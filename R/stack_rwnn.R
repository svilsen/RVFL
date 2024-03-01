#############################################################################
####################### Stacking ERWNN neural networks #######################
#############################################################################

#' @title Stacking random weight neural networks
#' 
#' @description Use stacking to create ensemble random weight neural networks.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of each RWNN
#' @param B The number of models in the stack.
#' @param optimise TRUE/FALSE: Should the stacking weights be optimised (or should the stack just predict the average)? 
#' @param folds The number of folds used to train the RWNN models. 
#' @param method The penalisation type passed to \link{ae_rwnn}. Set to \code{NULL} (default), \code{"l1"}, or \code{"l2"}. If \code{NULL}, the \link{rwnn} is used as the base learner.
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
stack_rwnn <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, method = NULL, type = NULL, control = list()) {
    UseMethod("stack_rwnn")
}

stack_rwnn.matrix <- function(X, y, N_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, method = NULL, type = NULL, control = list()) {
    ## Checks
    
    if (is.null(control[["include_data"]])) {
        control$include_data <- FALSE
    }
    
    dc <- data_checks(y, X)
    
    if (!is.logical(optimise)) {
        stop("'optimise' has to be 'TRUE'/'FALSE'.")
    }
    
    if (optimise) {
        if (is.null(folds) || folds < 1) {
            folds <- 10
            warning("Note: 'folds' was not supplied, and is set to 10.")
        }
    } 
    else {
        folds <- 1
    }
    
    if (is.null(B) | !is.numeric(B)) {
        B <- 100
        warning("Note: 'B' was not supplied, 'B' was set to 100.")
    }
    
    if (is.null(control$N_features)) {
        control$N_features <- ceiling(dim(X)[2] / 3)
    }
    
    control$N_hidden <- N_hidden
    control <- do.call(control_rwnn, control)
    
    ##
    if (optimise) {
        fold_index <- create_folds(X, folds)
        C <- matrix(NA, nrow = nrow(X), ncol = B)
    }
    
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        if (!is.null(method)) {
            object_b <- rwnn.matrix(X = X, y = y, N_hidden = N_hidden, lambda = lambda, type = type, control = control)
        }
        else {
            object_b <- ae_rwnn.matrix(X = X, y = y, N_hidden = N_hidden, lambda = lambda, method = method, type = type, control = control)
        }
        
        if (optimise) {
            H <- rwnn_forward(X, object_b$Weights$Hidden, object_b$activation, object_b$Bias$Hidden)
            H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = N_hidden[i]))
            
            if (object_b$Combined$Hidden) {
                H <- do.call("cbind", H)
            } else {
                H <- H[[length(H)]]
            }
            
            if (object_b$Bias$Output) {
                H <- cbind(1, H)
            }
            
            O <- H
            if (object_b$Combined$Input) {
                O <- cbind(X, H)
            }
            
            for (k in seq_len(folds)) {
                Ok <- matrix(O[-fold_index[[k]], ], ncol = ncol(O))
                Om <- matrix(O[fold_index[[k]], ], ncol = ncol(O))
                yk <- matrix(y[-fold_index[[k]], ], ncol = ncol(y))
                beta_b <- estimate_output_weights(Ok, yk, control$lnorm, lambda)$beta
                
                C[fold_index[[k]], b] <- Om %*% beta_b
            }
        }
        
        objects[[b]] <- object_b
    }
    
    ##
    if (optimise) {
        w <- estimate_weights_stack(C = C, b = y, B = B)
    } else {
        w <- rep(1 / B, B)
    }
    
    ##
    object <- list(
        formula = NULL,
        data = list(X = X, y = y, C = ifelse(type == "regression", NA, colnames(y))), 
        RWNNmodels = objects, 
        weights = w, 
        method = "stacking"
    )  
    
    class(object) <- "ERWNN"
    return(object)
}

#' @rdname stack_rwnn
#' @method stack_rwnn formula
#' 
#' @example inst/examples/stackrwnn_example.R
#' 
#' @export
stack_rwnn.formula <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, method = NULL, type = NULL, control = list()) {
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
    y <- as.matrix(y, nrow = nrow(data))
    
    #
    if (is.null(type)) {
        if (class(y[, 1]) == "numeric") {
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
    mm <- stack_rwnn.matrix(X, y, N_hidden = N_hidden, lambda = lambda, B = B, optimise = optimise, folds = folds, method = method, type = type, control = control)
    mm$formula <- formula
    return(mm)
}
