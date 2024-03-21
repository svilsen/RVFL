#' @keywords internal 
"_PACKAGE"

#' @title RWNN: Random Weight Neural Networks
#'
#' @author Søren B. Vilsen <svilsen@math.aau.dk>
#' 
#' @importFrom Rcpp evalCpp
#' 
#' @importFrom stats coef cor predict runif sd rnorm pnorm dnorm quantile terms model.matrix model.response model.frame as.formula delete.response princomp
#' 
#' @importFrom graphics plot abline boxplot hist
#' 
#' @importFrom grDevices dev.hold dev.flush
#' 
#' @importFrom quadprog solve.QP
#' 
#' @importFrom utils methods
#' 
#' @importFrom methods formalArgs is
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
#'     \item{\code{n_hidden}}{The vector of neurons in each layer.}
#'     \item{\code{activation}}{The vector of the activation functions used in each layer.}
#'     \item{\code{lnorm}}{The norm used when estimating the output weights.}
#'     \item{\code{lambda}}{The penalisation constant used when estimating the output weights.}
#'     \item{\code{bias}}{The \code{TRUE/FALSE} bias vectors set by the control function for both hidden layers, and the output layer.}
#'     \item{\code{weights}}{The weigths of the neural network, split into random (stored in hidden) and estimated (stored in output) weights.}
#'     \item{\code{sigma}}{The standard deviation of the corresponding linear model.}
#'     \item{\code{type}}{A string indicating the type of modelling problem.}
#'     \item{\code{combined}}{A list of two \code{TRUE/FALSE} values stating whether the direct links were made to the input, and whether the output of each hidden layer was combined to make the prediction.}
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
#'     \item{\code{models}}{A list with each element being an \link{RWNN-object}.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method.}
#' }
#' 
#' @name ERWNN-object
#' @rdname ERWNN-object
NULL

#' @title Example data
#' 
#' @description A data-set of 2000 observations were sampled independently according to the function: 
#' \deqn{y_n = \dfrac{1}{1 + \exp(-x_n^T\beta + \varepsilon_n)},}
#' where \eqn{x_n^T} is a vector containing an intercept and five input features, \eqn{\beta} is a vector containing the parameters, \eqn{(-1\;2\;1\;2\;0.5\;3)^T}, and \eqn{\varepsilon_n} is normally distributed noise with mean 0 and variance 0.1. Furthermore, the five features were generated as \eqn{x_1 \sim \mathcal Unif(-5, 5)}, \eqn{x_2 \sim \mathcal Unif(0, 2)}, \eqn{x_3 \sim \mathcal N(2, 4)}, \eqn{x_4 \sim \mathcal Gamma(2, 4)}, and \eqn{x_5 \sim \mathcal Beta(10, 4)}, respectively.
#' 
"example_data"
