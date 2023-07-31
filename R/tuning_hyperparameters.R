######################################################################
####################### Tuning hyperparameters #######################
######################################################################

#' @title Hyper-parameter tuning
#' 
#' @description Simple function for hyper-parameter tuning using k-fold cross-validation.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param method The RWNN method in need of hyper-parameter optimisation.
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param folds The number of folds used in k-fold cross-validation.
#' @param hyperparameters A list of sequences of hyper-parameters to be tested.
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' @param trace A numeric indicating how often a trace should be shown (default is 0).
#' 
#' @example inst/examples/tuning_example.R
#' 
#' @return Either an \link{RWNN-object} or \link{ERWNN-object}.
#' 
#' @export
tune_hyperparameters <- function(formula, method, data = NULL, folds = 10, hyperparameters = list(), control = list(), trace = 0) {
    UseMethod("tune_hyperparameters")
}

tune_hyperparameters.matrix <- function(X, y, method, folds = 10, hyperparameters = list(), control = list(), trace = 0) {
    ## Checks
    #
    if (!is.function(method)) {
        stop("'method' has to be a function.")
    } else {
        method_name <- suppressWarnings(methods(method))
        if (!grepl(pattern = "rwnn", x = method_name[1])) {
            stop("The tuning function is only implemented for 'RWNN' and 'ERWNN' methods.")
        }
    }
    
    #
    dc <- data_checks(y, X)
    
    #
    if (is.null(folds) | !is.numeric(folds)) {
        stop("'folds' was either 'NULL' or not numeric.") 
    }
    
    if (folds < 1) {
        folds <- 3
        warning("'folds' was smaller than 1, setting 'folds' to 3.")
    } else if (folds == 1) {
        warning("'folds' was equal to 1, this is not recommended, as no validation set is generated.")
    } else if (folds > nrow(X)) {
        folds <- nrow(X)
        warning("'folds' was larger than the number of observations, setting 'folds' equal to the number of observations.")
    }
    
    if (is.null(trace) | !is.numeric(trace)) {
        trace <- 0
        warning("'trace' was either 'NULL' or not numeric, setting 'trace' to 0.") 
    }
    
    # 
    method_args <- names(formals(method))
    method_args <- method_args[-which(method_args %in% c("X", "y", "formula", "data", "control"))]
    hyperparameters_args <- names(hyperparameters)
    
    if (!all(method_args %in% hyperparameters_args)) { 
        missing_args <- method_args[!(method_args %in% hyperparameters_args)]
        stop(paste("Missing arguments in 'hyperparameters':", paste(missing_args, collapse = ", ")))
    }
    
    ##
    best_model_error <- Inf
    best_model_parameters <- NULL
    
    folds_index <- create_folds(X, folds)
    search_grid <- expand.grid(hyperparameters)
    for (i in seq_len(nrow(search_grid))) {
        if (trace > 0 && ((i == 1) || (i == nrow(search_grid)) || ((i %% trace) == 0))) {
            cat("Search parameters:", i, "/", nrow(search_grid), "\n")
        }
        
        model_error_i <- 0
        model_parameters_i <- search_grid[i, ]
        
        for (j in seq_along(folds_index)) {
            ##
            if (folds > 1) {
                X_train_i <- matrix(X[-folds_index[[j]], ], ncol = ncol(X))
                y_train_i <- matrix(y[-folds_index[[j]], ], ncol = ncol(y))
                
                X_val_i <- matrix(X[folds_index[[j]], ], ncol = ncol(X))
                y_val_i <- matrix(y[folds_index[[j]], ], ncol = ncol(y))
            } else {
                X_train_i <- X
                y_train_i <- y
                
                X_val_i <- X
                y_val_i <- y
            }
            
            ##
            model_args_ij <- list(formula = y_train_i ~ X_train_i, control = control)
            model_args_ij <- append(model_args_ij, model_parameters_i)
            
            if (is.list(model_args_ij$N_hidden)) {
                model_args_ij$N_hidden <- model_args_ij$N_hidden[[1]]
            }
            
            ##
            model_ij <- suppressWarnings(do.call(method, model_args_ij))
            
            ##
            model_error_ij <- mse(model_ij, X_val_i, y_val_i) 
            model_error_i <- model_error_i + model_error_ij
        }
        
        if (model_error_i < best_model_error) {
            best_model_error <- model_error_i
            best_model_parameters <- model_parameters_i
        } 
    }
    
    ##
    return(best_model_parameters)
}


#' @rdname tune_hyperparameters
#' @method tune_hyperparameters formula
#' 
#' @export
tune_hyperparameters.formula <- function(formula, method, data = NULL, folds = 10, hyperparameters = list(), control = list(), trace = 0) {
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
    best_model_parameters <- tune_hyperparameters.matrix(
        X = X, y = y, method = method, folds = folds, hyperparameters = hyperparameters, 
        control = control, trace = trace
    )
    
    model_best_args <- list(formula = formula, data = data, control = control)
    model_best_args <- append(model_best_args, best_model_parameters)
    
    if (is.list(model_best_args$N_hidden)) {
        model_best_args$N_hidden <- model_best_args$N_hidden[[1]]
    }
    
    mm <- do.call(method, model_best_args)
    mm$formula <- formula
    return(mm)
}

