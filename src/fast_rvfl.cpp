#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

// 
arma::mat sigmoid(const arma::mat & x) {
    return 1.0 / (1.0 + arma::exp(-1.0 * x));
}

arma::mat bind_cols(arma::mat A, arma::mat B){
    arma::mat M = join_horiz(A, B); 
    return M;
}

// 
//[[Rcpp::export]]
arma::mat rvfl_forward(arma::mat X, 
                       const std::vector<arma::mat> & W, 
                       const std::vector<bool> & bias) {
    const unsigned int & N = X.n_rows;
    const unsigned int & M = W.size();
    
    arma::colvec b(N, arma::fill::ones);
    if (bias[0]) {
        X = bind_cols(b, X);
    }
    
    arma::mat H = X * W[0];
    for (unsigned int m = 1; m < M; m++) {
        if (bias[m]) {
            H = bind_cols(b, H);
        }
        
        H = H * W[m];
        H = sigmoid(H);
    } 
    
    return H;
}

//[[Rcpp::export]]
Rcpp::List estimate_output_weights(const arma::mat & O, const arma::colvec & y) {
    const int & N = O.n_rows;
    const int & p = O.n_cols;
    
    const arma::mat OO = arma::pinv(arma::trans(O) * O);
    
    const arma::colvec beta = OO * arma::trans(O) * y; 
    const arma::colvec residual = y - O * beta; 
    
    const double sigma_squared = arma::as_scalar(arma::trans(residual) * residual / (N - p));
    const arma::colvec standard_error = arma::sqrt(sigma_squared * arma::diagvec(OO));
    
    return Rcpp::List::create(
        Rcpp::Named("beta") = beta,
        Rcpp::Named("se") = standard_error
    );
}

