###################################################
####################### AUX #######################
###################################################

data_checks <- function(y, X) {
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
    
    return(NULL)
}

create_folds <- function(X, folds) {
    N <- nrow(X)
    index <- sample(N, N, replace = FALSE)
    fold_index <- rep(seq_len(folds), each = floor(N / folds))
    
    if (length(fold_index) < length(index)) {
        fold_index <- c(fold_index, seq_len(folds)[seq_len(length(index) - length(fold_index))])
    }
    
    return(unname(split(x = index, f = fold_index)))
}

strip_terms <- function(formula) {
    attr_names <- names(attributes(formula))
    for (i in seq_along(attr_names)) {
        attr(formula, attr_names[i]) <- NULL
    }
    
    formula <- as.formula(formula)
    return(formula)
}

#' @name Errors
#' @rdname errors 
#' 
#' @title Error functions 
#' 
#' @description Simple error functions for finding MSE, RMSE, MAE, and MAPE. 
#' 
#' @param object A model object.
#' @param X A matrix of observed features.
#' @param y A vector of observed targets.
#' 
#' @return Error for the provided data-set.
NULL 

#' @rdname errors
#' @export  
mse <- function(object, X, y) {
    yhat <- predict(object, newdata = X)
    return(mean((y - yhat)^2))
}

#' @rdname errors
#' @export  
rmse <- function(object, X, y) {
    return(sqrt(mse(object, X, y)))
}

#' @rdname errors
#' @export  
mae <- function(object, X, y) {
    yhat <- predict(object, newdata = X)
    return(mean(abs(y - yhat)))
}

#' @rdname errors
#' @export  
mape <- function(object, X, y) {
    yhat <- predict(object, newdata = X)
    return(mean(abs((y - yhat) / y)))
}
