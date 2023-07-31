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
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
stack_rwnn <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, control = list()) {
    UseMethod("stack_rwnn")
}

stack_rwnn.matrix <- function(X, y, N_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, control = list()) {
    ## Checks
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
        object_b <- rwnn.matrix(X = X, y = y, N_hidden = N_hidden, lambda = lambda, control = control)
        
        if (optimise) {
            H <- rwnn_forward(X, object_b$Weights$Hidden, object_b$activation, object_b$Bias$Hidden)
            H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = N_hidden[i]))
            H <- do.call("cbind", H)
            
            if (object_b$Bias$Output) {
                H <- cbind(1, H)
            }
            
            O <- H
            if (object_b$Combined) {
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
        data = list(X = X, y = y), 
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
stack_rwnn.formula <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, B = 100, optimise = FALSE, folds = 10, control = list()) {
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
    mm <- stack_rwnn.matrix(X, y, N_hidden = N_hidden, lambda = lambda, B = B, optimise = optimise, folds = folds, control = control)
    mm$formula <- formula
    return(mm)
}
