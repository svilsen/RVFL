#' @title RVFL: Random Vector Functional Link Neural Network Models
#'
#' @description Creation, estimation, and prediction of random vector functional link neural networks (RVFL), including popular variants like extreme learning machines (ELM), sparse RVFL (spRVFL), and deep RVFL (dRVFL). It further allows for the creation of ensemble RVFLs like bagging RVFL (bagRVFL), boosting RVFL (boostRVFL), stacking RVFL (stackRVFL), and ensemble deep RVFL (edRVFL).
#' 
#' @docType package
#' 
#' @author Søren B. Vilsen <svilsen@math.aau.dk>
#' 
#' @importFrom Rcpp evalCpp
#' 
#' @importFrom stats coef predict runif sd rnorm dnorm quantile
#' 
#' @importFrom graphics plot abline boxplot hist
#' 
#' @importFrom grDevices dev.hold dev.flush
#' 
#' @importFrom quadprog solve.QP
#' 
#' @importFrom utils methods
#' 
#' @importFrom methods formalArgs
#' 
#' @useDynLib RVFL
#' 
#' @name RVFL-package
#' 
#' @rdname RVFL-package
NULL

#' @title An RVFL-object 
#' 
#' @description An RVFL-object is a list containing the following:
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
#' @name RVFL-object
#' @rdname RVFL-object
NULL

#' @title An ERVFL-object 
#' 
#' @description An ERVFL-object is a list containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list with each element being an \link{RVFL-object}.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method.}
#' }
#' 
#' @name ERVFL-object
#' @rdname ERVFL-object
NULL

#' @title An SRVFL-object 
#' 
#' @description An SRVFL-object is a list containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list with each element being an \link{RVFL-object}.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method.}
#' }
#' 
#' @name SRVFL-object
#' @rdname SRVFL-object
NULL
