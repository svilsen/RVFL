#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

arma::mat bind_cols(const arma::mat & A, const arma::mat & B){
    arma::mat M = join_horiz(A, B); 
    return M;
}