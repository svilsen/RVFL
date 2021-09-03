#############################################################################
####################### Stackig ERVFL neural networks #######################
#############################################################################

#' @title Stacking random vector functional links
#' 
#' @description Use stacking to create ensemble random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param B The number of models in the stack.
#' @param lambda The penalisation constant used when training the output layers of each RVFL.
#' @param optimise TRUE/FALSE: Should the stacking weights be optimised (or should the stack just predict the average)? 
#' @param folds The number of folds used to train the RVFL models. 
#' @param control A list of additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An \link{ERVFL-object}.
#' 
#' @export
stackRVFL <- function(X, y, N_hidden, B = 100, lambda = 0, optimise = FALSE, folds = 10, control = list()) {
    UseMethod("stackRVFL")
}

#' @rdname stackRVFL
#' @method stackRVFL default
#' 
#' @example inst/examples/stackrvfl_example.R
#' 
#' @export
stackRVFL.default <- function(X, y, N_hidden, B = 100, lambda = 0, optimise = FALSE, folds = 10, control = list()) {
    ## Checks
    dc <- data_checks(y, X)
    
    if (optimise && (is.null(folds) || folds < 1)) {
        folds <- 10
    }
    
    if (!optimise) {
        folds <- 1
    }
    
    if (is.null(B)) {
        B <- 100
        warning(paste0("Note: 'B' was not supplied, 'B' was set to ", B, "."))
    }
    
    if (is.null(control$N_features)) {
        control$N_features <- ceiling(dim(X)[2] / 3)
    }
    
    ##
    if (optimise) {
        fold_index <- create_folds(X, folds)
    }
    
    objects <- vector("list", B)
    for (b in seq_len(B)) {
        object_b <- RVFL(X = X, y = y, N_hidden = N_hidden, lambda = lambda, control = control)
        
        beta_b <- vector("list", folds)
        for (k in seq_len(folds)) {
            if (optimise) {
                Xk <- matrix(X[-fold_index[[k]], ], ncol = ncol(X))
                yk <- matrix(y[-fold_index[[k]], ], ncol = ncol(y))
            } else {
                Xk <- X
                yk <- y
            }
            
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
        
        object_b$Weights$Output <- matrix(apply(do.call("cbind", beta_b), 1, mean), ncol = 1)
        objects[[b]] <- object_b
    }
    
    ##
    if (optimise) {
        C <- do.call("cbind", lapply(objects, predict))
        w <- estimate_weights_stack(C = C, b = y, B = B)
    } else {
        w <- rep(1 / B, B)
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
