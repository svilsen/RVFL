#ifndef rwnn_forward
#define rwnn_forward

#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

std::vector<arma::mat> rwnn_forward(
        arma::mat X, 
        const std::vector<arma::mat> & W, 
        const std::vector<std::string> & activation,
        const std::vector<bool> & bias
);

#endif //rwnn_forward