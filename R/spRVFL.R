######################################################################################
####################### Sparse pre-trained RVFL neural network #######################
######################################################################################

#' @title Random vector functional link
#' 
#' @description Set-up and estimate weights of a random vector functional link neural network.
#' 
#' @param X A matrix of observed features used to train the parameters of the output layer.
#' @param y A vector of observed targets used to train the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layer.
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @details The function \code{ELM} is a wrapper for the general \code{RVFL} function without the link between features and targets. Furthermore, notice that \code{dRVFL} is handled by increasing the number of elements passed in \code{N_hidden}.
#' 
#' @return An RVFL-object containing the random and fitted weights of the RVFL-model. An RVFL-object contains the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{N_hidden}}{The vector of neurons in each layer.}
#'     \item{\code{activation}}{The vector of the activation functions used in each layer.}
#'     \item{\code{Bias}}{The \code{TRUE/FALSE} bias vectors set by the control function for both hidden layers, and the output layer.}
#'     \item{\code{Weights}}{The weigths of the neural network, split into random (stored in hidden) and estimated (stored in output) weights.}
#'     \item{\code{Sigma}}{The standard deviation of the corresponding linear model.}
#'     \item{\code{Combined}}{A \code{TRUE/FALSE} stating whether the direct links were made to the input.}
#' }
#' 
#' @export
spRVFL <- function(X, y, N_hidden, lambda = 0, ...) {
    UseMethod("spRVFL")
}

#' @rdname spRVFL
#' @method spRVFL default
#' 
#' @example inst/examples/rvfl_example.R
#' 
#' @export
spRVFL.default <- function(X, y, N_hidden, lambda = 0, ...) {
    return(NULL)
}