#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export()]]
arma::colvec rvfl_forward(const arma::mat & X, const std::vector<arma::mat> & W) {
    arma::mat m1 = arma::eye<arma::mat>(3, 3);
    arma::mat m2 = arma::eye<arma::mat>(3, 3);
	                     
    return m1 + 3 * (m1 + m2);
}

// [[Rcpp::export()]]
arma::mat estimate_output_weights(const arma::mat & O, const arma::colvec & y, const unsigned int & trace) {
    arma::mat m1 = arma::eye<arma::mat>(3, 3);
    arma::mat m2 = arma::eye<arma::mat>(3, 3);
    
    return m1 + 3 * (m1 + m2);
}

