###################################################
####################### AUX #######################
###################################################

create_folds <- function(X, folds) {
    index <- sample(nrow(X), nrow(X), replace = FALSE)
    fold_index <- rep(seq_len(folds), each = floor(nrow(X) / folds))
    
    if (length(fold_index) < length(index)) {
        fold_index <- c(fold_index, seq_len(folds)[seq_len(length(index) - length(fold_index))])
    }
    
    return(unname(split(x = index, f = fold_index)))
}

#' @name Errors
#' @rdname errors 
#' 
#' @title Error functions 
#' 
#' @description Simple error functions for finding MSE, RMSE, MAE, and MAPE. 
#' 
#' @param object A model object
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
