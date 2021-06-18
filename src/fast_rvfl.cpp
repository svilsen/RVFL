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
arma::mat rvfl_forward(arma::mat X, 
                       const std::vector<arma::mat> & W, 
                       const std::vector<std::string> & activation,
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
        if (activation[m] == "sigmoid") {
            H = sigmoid(H);
        }
        else if (activation[m] == "tanh") {
            H = tanh(H);
        }
        else if (activation[m] == "relu") {
            H = relu(H);
        }
        else if (activation[m] == "silu") {
            H = silu(H);
        }
        else if (activation[m] == "softplus") {
            H = softplus(H);
        }
        else if (activation[m] == "softsign") {
            H = softsign(H);
        }
        else if (activation[m] == "sqnl") {
            H = sqnl(H);
        }
        else if (activation[m] == "gaussian") {
            H = gaussian(H);
        }
        else if (activation[m] == "sqrbf") {
            H = sqrbf(H);
        }
        else if (activation[m] == "bentidentity") {
            H = bentidentity(H);
        }
        else if (activation[m] == "identity") {
            H = identity(H);
        }
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
        Rcpp::Named("sigma") = std::sqrt(sigma_squared),
        Rcpp::Named("se") = standard_error
    );
}

