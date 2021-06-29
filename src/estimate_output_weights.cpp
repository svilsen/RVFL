#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

//[[Rcpp::export]]
Rcpp::List estimate_output_weights(const arma::mat & O, const arma::colvec & y, const double & lambda) {
    const int & N = O.n_rows;
    const int & p = O.n_cols;
    
    arma::mat Op;
    if (lambda > 1e-8) {
        const arma::mat lambdaI = lambda * arma::eye(p, p);
        const arma::mat OO = arma::trans(O) * O;
        const arma::mat OI = arma::inv(OO + lambdaI);
        
        Op = OI * arma::trans(O);
    }
    else{
        Op = arma::pinv(O); 
    }
    
    const arma::colvec beta = Op * y;  
    const arma::colvec residual = y - O * beta; 
    
    const double sigma_squared = arma::as_scalar(arma::trans(residual) * residual / (N - p));
    return Rcpp::List::create(
        Rcpp::Named("beta") = beta,
        Rcpp::Named("sigma") = std::sqrt(sigma_squared)
    );
}
