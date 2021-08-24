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
#' @importFrom stats coef predict runif sd
#' 
#' @importFrom Rsolnp solnp
#' 
#' @importFrom graphics plot abline
#' 
#' @importFrom grDevices dev.hold dev.flush
#' 
#' @importFrom quadprog solve.QP
#' 
#' @useDynLib RVFL
#' 
#' @name RVFL-package
#' 
#' @rdname RVFL-package
NULL
