#' @title RWNN: Random Weight Neural Networks
#'
#' @description Creation, estimation, and prediction of random weight neural networks (RWNN), including popular variants like extreme learning machines (ELM), sparse RWNN (spRWNN), and deep RWNN (dRWNN). It further allows for the creation of ensemble RWNNs like bagging RWNN (bagRWNN), boosting RWNN (boostRWNN), stacking RWNN (stackRWNN), and ensemble deep RWNN (edRWNN).
#' 
#' @docType package
#' 
#' @author Søren B. Vilsen <svilsen@math.aau.dk>
#' 
#' @importFrom Rcpp evalCpp
#' 
#' @importFrom stats coef predict runif sd rnorm dnorm quantile terms model.matrix model.response model.frame as.formula delete.response
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
#' @importFrom randtoolbox halton sobol
#' 
#' @useDynLib RWNN
#' 
#' @name RWNN-package
#' 
#' @rdname RWNN-package
NULL

#' @title An RWNN-object 
#' 
#' @description An RWNN-object is a list containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{N_hidden}}{The vector of neurons in each layer.}
#'     \item{\code{activation}}{The vector of the activation functions used in each layer.}
#'     \item{\code{lnorm}}{The norm used when estimating the output weights.}
#'     \item{\code{lambda}}{The penalisation constant used when estimating the output weights.}
#'     \item{\code{Bias}}{The \code{TRUE/FALSE} bias vectors set by the control function for both hidden layers, and the output layer.}
#'     \item{\code{Weights}}{The weigths of the neural network, split into random (stored in hidden) and estimated (stored in output) weights.}
#'     \item{\code{Sigma}}{The standard deviation of the corresponding linear model.}
#'     \item{\code{Type}}{A string indicating the type of modelling problem.}
#'     \item{\code{Combined}}{A list of two \code{TRUE/FALSE} values stating whether the direct links were made to the input, and whether the output of each hidden layer was combined to make the prediction.}
#' }
#' 
#' @name RWNN-object
#' @rdname RWNN-object
NULL

#' @title An ERWNN-object 
#' 
#' @description An ERWNN-object is a list containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RWNNmodels}}{A list with each element being an \link{RWNN-object}.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method.}
#' }
#' 
#' @name ERWNN-object
#' @rdname ERWNN-object
NULL

#' @title Example data
#' 
#' @description Data generated using a simple non-linear function with 5 inputs and 1 output.
"example_data"
