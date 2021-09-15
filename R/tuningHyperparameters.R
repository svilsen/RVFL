######################################################################
####################### Tuning hyperparameters #######################
######################################################################

#' @title Random vector functional link
#' 
#' @description Set-up and estimate weights of a random vector functional link neural network.
#' 
#' @param method 
#' @param X A matrix of observed features used to train the parameters of the output layer.
#' @param y A vector of observed targets used to train the parameters of the output layer.
#' @param folds The number of folds used in k-fold cross-validation.
#' @param control A list of additional arguments passed to the \link{control_RVFL} function.
#' @param ... 
#' 
#' @return An object either of \link{RVFL} or \link{ERVFL}.
#' 
#' @export
tune_hyperparameters <- function(method, X, y, folds = 10, control = list(), ...) {
    
}