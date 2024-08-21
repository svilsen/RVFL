#ifndef estimate_output_weights
#define estimate_output_weights

#include <RcppArmadillo.h>

//[[Rcpp::depends(RcppArmadillo)]]

double rho(const arma::mat & O, const arma::colvec & y, const arma::colvec & beta, const int & j);

arma::colvec coordinate_descent(const arma::mat & O, const arma::colvec & y, 
                                const double & lambda, const arma::colvec & beta0, 
                                const int & N, const int & p, const bool & trace);

Rcpp::List estimate_output_weights(const arma::mat & O, const arma::colvec & y, const std::string & lnorm, const double & lambda);

#endif //estimate_output_weights