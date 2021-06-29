#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

// Auxilliary functions
arma::mat bind_cols(const arma::mat & A, const arma::mat & B){
    arma::mat M = join_horiz(A, B); 
    return M;
}

// Activation functions 
arma::mat sigmoid(const arma::mat & x) {
    return 1.0 / (1.0 + arma::exp(-1.0 * x));
}

arma::mat tanh(const arma::mat & x) {
    return arma::tanh(x);
}

arma::mat relu(const arma::mat & x) {
    int N = x.n_rows;
    int M = x.n_cols;
    
    arma::mat y = x;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            if (y(n, m) < 0.0) {
                y(n, m) = 0.0;
            }
        }
    }
    
    return y;
}

arma::mat silu(const arma::mat & x) {
    return x / (1.0 + arma::exp(-1.0 * x));
}

arma::mat softplus(const arma::mat & x) {
    return arma::log(1.0 + arma::exp(x));
}

arma::mat softsign(const arma::mat & x) {
    return x / (1.0 + arma::abs(x));
}

arma::mat sqnl(const arma::mat & x) {
    int N = x.n_rows;
    int M = x.n_cols;
    
    arma::mat y = x;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            if (y(n, m) < -2.0) {
                y(n, m) = -1.0;
            }
            else if (y(n, m) < 0.0) {
                y(n, m) = y(n, m) + 0.25 * y(n, m) * y(n, m);
            }
            else if (y(n, m) < 2.0) {
                y(n, m) = y(n, m) - 0.25 * y(n, m) * y(n, m);
            }
            else {
                y(n, m) = 1.0;
            }
        }
    }
    
    return y;
}

arma::mat gaussian(const arma::mat & x) {
    return arma::exp(-1.0 * x % x);
}

arma::mat sqrbf(const arma::mat & x) {
    int N = x.n_rows;
    int M = x.n_cols;
    
    arma::mat y = x;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            if (std::abs(y(n, m)) < 1.0) {
                y(n, m) = 1.0 - 0.50 * y(n, m) * y(n, m);
            }
            else if (std::abs(y(n, m)) < 2.0) {
                y(n, m) = y(n, m) + 0.25 * y(n, m) * y(n, m);
            }
            else {
                y(n, m) = 0.0;
            }
        }
    }
    
    return y;
}

arma::mat bentidentity(const arma::mat & x) {
    return 0.5 * (arma::exp(0.5 * arma::log(x % x + 1.0)) - 1.0) + x;
}

arma::mat identity(const arma::mat & x) {
    return x;
}

// RVFL functions
//[[Rcpp::export]]
std::vector<arma::mat> rvfl_forward(arma::mat X, 
                                    const std::vector<arma::mat> & W, 
                                    const std::vector<std::string> & activation,
                                    const std::vector<bool> & bias) {
    const unsigned int & N = X.n_rows;
    const unsigned int & M = W.size();
    
    arma::colvec b(N, arma::fill::ones);
    if (bias[0]) {
        X = bind_cols(b, X);
    }
    
    std::vector<arma::mat> H(M); 
    H[0] = X * W[0];
    for (unsigned int m = 1; m < M; m++) {
        arma::mat H_m = H[m - 1];
        
        if (bias[m]) {
            H_m = bind_cols(b, H_m);
        }
        
        H_m = H_m * W[m];
        if (activation[m] == "sigmoid") {
            H_m = sigmoid(H_m);
        }
        else if (activation[m] == "tanh") {
            H_m = tanh(H_m);
        }
        else if (activation[m] == "relu") {
            H_m = relu(H_m);
        }
        else if (activation[m] == "silu") {
            H_m = silu(H_m);
        }
        else if (activation[m] == "softplus") {
            H_m = softplus(H_m);
        }
        else if (activation[m] == "softsign") {
            H_m = softsign(H_m);
        }
        else if (activation[m] == "sqnl") {
            H_m = sqnl(H_m);
        }
        else if (activation[m] == "gaussian") {
            H_m = gaussian(H_m);
        }
        else if (activation[m] == "sqrbf") {
            H_m = sqrbf(H_m);
        }
        else if (activation[m] == "bentidentity") {
            H_m = bentidentity(H_m);
        }
        else if (activation[m] == "identity") {
            H_m = identity(H_m);
        }
        
        H[m] = H_m;
    } 
    
    return H;
}


