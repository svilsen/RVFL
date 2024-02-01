#ifndef fista
#define fista

#include <RcppArmadillo.h>

double f(const arma::mat & X, const arma::mat & H, const arma::mat & B);

arma::mat f_grad(const arma::mat & X, const arma::mat & H, const arma::mat & B);

double L1(const arma::mat & B);

arma::mat L1_proxy(const arma::mat & B, const double & tau);

Rcpp::List fista(
        const arma::mat & X, const arma::mat & H, const arma::mat & W, const double & tau, const int & max_iterations, 
        const int & w, const double & step_shrink, const int & backtrack, const double & tolerance, const int & trace
);

#endif //fista