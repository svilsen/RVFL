#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

double rss(const arma::mat & O, const arma::colvec & y, const double & lambda, const arma::colvec & beta) {
    const arma::colvec residual = y - O * beta;
    const double rss = arma::as_scalar(arma::trans(residual) * residual);
    const double l1 = arma::sum(arma::abs(beta));
    return rss + lambda * l1;
}

double rho(const arma::mat & O, const arma::colvec & y, const arma::colvec & beta, const int & j) {
    arma::colvec beta_j = beta;
    beta_j[j] = 0.0; 
    
    const arma::colvec residual = (y - O * beta_j);
    const double z = arma::as_scalar(arma::trans(O.col(j)) * residual);
    return z;
}

arma::colvec coordinate_descent(const arma::mat & O, const arma::colvec & y, 
                                const double & lambda, const arma::colvec & beta0, 
                                const int & N, const int & p, const bool & trace) {
    arma::colvec z(p, arma::fill::ones);
    for (int n = 0; n < N; n++) {
        const arma::colvec o = arma::trans(O.row(n));
        z += o % o;
    }
    
    int i = 0;
    bool converged = false;
    double e_old, e_new = HUGE_VAL;
    arma::colvec beta = beta0;
    while (!converged) {
        e_old = e_new;
        e_new = 0.0;
        for (int j = 0; j < p; j++) {
            double beta_j = beta[j];
            double rho_j = rho(O, y, beta, j);
            if (rho_j < (-lambda / 2.0)) {
                beta[j] = (rho_j + lambda / 2.0) / z[j];
            }
            else if (rho_j > (lambda / 2.0)) {
                beta[j] = (rho_j - lambda / 2.0) / z[j];
            }
            else {
                beta[j] = 0.0;
            }
            
            if (std::abs(beta[j] - beta_j) > e_new) {
                e_new = std::abs(beta[j] - beta_j);
            }
        }
        
        converged = (std::abs(e_old - e_new) < 1e-8) || (i > 1000);
        i++;
    }
    
    return beta;
}

//[[Rcpp::export]]
Rcpp::List estimate_output_weights(const arma::mat & O, const arma::colvec & y, const std::string & lnorm, const double & lambda) {
    const int & N = O.n_rows;
    const int & p = O.n_cols;
    
    arma::colvec beta;
    if (lambda < 1e-8) {
        const arma::mat Op = arma::pinv(O); 
        beta = Op * y;  
    }
    else if (lnorm == "l2") {
        const arma::mat Op = arma::inv(arma::trans(O) * O + lambda * arma::eye(p, p)) * arma::trans(O);
        beta = Op * y;  
    }
    else {
        const arma::mat Op = arma::pinv(O); 
        const arma::colvec beta0 = Op * y;  
        
        beta = coordinate_descent(O, y, lambda, beta0, N, p, true);  
    }
    
    const arma::colvec residual = y - O * beta; 
    const double sigma_squared = arma::as_scalar(arma::trans(residual) * residual / (N - p));
    return Rcpp::List::create(
        Rcpp::Named("beta") = beta,
        Rcpp::Named("sigma") = std::sqrt(sigma_squared)
    );
}

