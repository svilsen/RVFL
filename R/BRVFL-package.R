#' @title BRVFL: Random Forest Random Vector Functional Link Neural Network Models
#'
#' @description Simple creation, estimation, and prediction of random vector functional link neural networks. Furthermore, it incorporates a random forest style boosting into the RVFL framework, yielding more robust estimation and prediction. 
#' 
#' @docType package
#' 
#' @author Søren B. Vilsen <svilsen@math.aau.dk>
#' 
#' @importFrom Rcpp evalCpp
#' 
#' @importFrom stats coef predict runif sd
#' 
#' @importFrom Rsolnp solnp
#' 
#' @useDynLib BRVFL
#' 
#' @name BRVFL
NULL