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
#'     \item{\code{lambda}}{The penalisation constant used when estimating the output weights.}
#'     \item{\code{Bias}}{The \code{TRUE/FALSE} bias vectors set by the control function for both hidden layers, and the output layer.}
#'     \item{\code{Weights}}{The weigths of the neural network, split into random (stored in hidden) and estimated (stored in output) weights.}
#'     \item{\code{Sigma}}{The standard deviation of the corresponding linear model.}
#'     \item{\code{Combined}}{A \code{TRUE/FALSE} stating whether the direct links were made to the input.}
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

#' @title An SRWNN-object 
#' 
#' @description An SRWNN-object is a list containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{N_hidden}}{The vector of neurons in each layer.}
#'     \item{\code{activation}}{The vector of the activation functions used in each layer.}
#'     \item{\code{lambda}}{The penalisation constant used when estimating the output weights.}
#'     \item{\code{Bias}}{The \code{TRUE/FALSE} bias vectors set by the control function for both hidden layers, and the output layer.}
#'     \item{\code{Samples}}{The sampled hidden weights and corresponding beta and sigma values for every sample.}
#'     \item{\code{Weights}}{The weigths of the neural network, split into random (stored in hidden) and estimated (stored in output) weights.}
#'     \item{\code{Sigma}}{The standard deviation of the corresponding linear model.}
#'     \item{\code{Combined}}{A \code{TRUE/FALSE} stating whether the direct links were made to the input.}
#' }
#' 
#' @name SRWNN-object
#' @rdname SRWNN-object
NULL

