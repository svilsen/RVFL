######################################################################
####################### Tuning hyperparameters #######################
######################################################################

#' @title Hyper-parameter tuning
#' 
#' @description Simple function for hyper-parameter tuning using k-fold cross-validation.
#' 
#' @param method 
#' @param X A matrix of observed features used to train the parameters of the output layer.
#' @param y A vector of observed targets used to train the parameters of the output layer.
#' @param folds The number of folds used in k-fold cross-validation.
#' @param hyperparameters A named list of hyper-parameters.
#' @param control A list of additional arguments passed to the \link{control_RVFL} function.
#' @param trace A numeric indicating how often a trace of should be shown (default is 0).
#' 
#' @return An object either of class \link{RVFL} or \link{ERVFL}.
#' 
#' @export
tune_hyperparameters <- function(method, X, y, folds = 10, hyperparameters = list(), control = list(), trace = 0) {
    ## Checks
    #
    if (!is.function(method)) {
        stop("'method' has to be a function.")
    } else {
        method_name <- suppressWarnings(methods(method))
        if (!grepl(pattern = "RVFL", x = method_name[1])) {
            stop("Only implemented for 'RVFL' and 'ERVFL' methods.")
        }
        
        if (grepl(pattern = "sampleRVFL", x = method_name[1])) {
            stop("Support for 'sampleRVFL' is not yet implemented.")
        }
    }
    
    #
    dc <- data_checks(y, X)
    
    #
    if (is.null(folds)) {
        folds <- 10
    }
    
    if (folds < 1) {
        folds <- 1
        warning("'folds' was smaller than 1... setting 'folds = 1'.") 
    } else if (folds > nrow(X)) {
        folds <- nrow(X)
        warning("'folds' was larger than the number of observations... setting 'folds = nrow(X)'.")
    }
    
    # 
    method_args <- names(formals(method))
    method_args <- method_args[-which(method_args %in% c("X", "y", "control"))]
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
            X_train_i <- matrix(X[-folds_index[[j]], ], ncol = ncol(X))
            y_train_i <- matrix(y[-folds_index[[j]], ], ncol = ncol(y))
            
            X_val_i <- matrix(X[folds_index[[j]], ], ncol = ncol(X))
            y_val_i <- matrix(y[folds_index[[j]], ], ncol = ncol(y))
            
            ##
            model_args_ij <- list(X = X_train_i, y = y_train_i, control = control)
            model_args_ij <- append(model_args_ij, model_parameters_i)
            
            if (is.list(model_args_ij$N_hidden)) {
                model_args_ij$N_hidden <- model_args_ij$N_hidden[[1]]
            }
            
            ##
            model_ij <- do.call(method, model_args_ij)
            
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
    model_best_args <- list(X = X, y = y, control = control)
    model_best_args <- append(model_best_args, best_model_parameters)
    
    if (is.list(model_best_args$N_hidden)) {
        model_best_args$N_hidden <- model_best_args$N_hidden[[1]]
    }
    
    model_best <- do.call(method, model_best_args)
    return(model_ij)
}








