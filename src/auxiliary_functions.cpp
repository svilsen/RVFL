#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

arma::mat matrix_sign(const arma::mat & M) {
    arma::mat S = M;
    
    for (auto & val : S) {
        if (val > 0) {
            val = 1.0;
        }
        else if (val < 0) {
            val = -1.0;
        }
        else {
            val = 0.0;
        }
    }
    
    return S;
}

arma::mat matrix_nonzero(const arma::mat & M) {
    arma::mat S = M;
    
    for (auto & val : S) {
        if (val < 0) {
            val = 0.0;
        }
    }
    
    return S;
}

bool matrix_condition(const arma::mat & M, const double & x) {
    bool s = false;
    for (auto & val : M) {
        if (x > val) {
            s = true;
            break;
        }
    }
    
    return s;
}

double max_window(const arma::mat & x, const int & w, const int & i) {
    int l = std::max(i - w, 0);
    int u = std::max(i - 1, 0);
    
    double max = x[l];
    for (int i = l; i < u; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    
    return max;
}

////
//[[Rcpp::export]]
arma::mat importance_score(const arma::mat &X, const arma::mat &W) {
    const int & N = W.n_rows;
    const int & M = W.n_cols;
    
    arma::mat Z = W;
    for (int n = 0; n < N; n++) {
        arma::colvec X_n = X.col(n);
        for (int m = 0; m < M; m++) {
            double W_nm = W(n, m);
            Z(n, m) = arma::accu(arma::abs(W_nm * X_n));
        }
    }
    
    return Z;
}

////
//[[Rcpp::export]]
std::vector<std::string> classify_cpp(const arma::mat &y, const std::vector<std::string> &C, const double &t, const double &b) {
    const int & N = y.n_rows;
    const int & d = y.n_cols;
    
    std::vector<std::string> yc(N);
    for (int n = 0; n < N; n++) {
        //
        const arma::rowvec & y_n = y.row(n);
        
        //
        int i = 0; 
        int j = 1; 
        
        //
        double m = y_n[i] - y_n[j]; 
        
        int s = 0; 
        if (y_n[i] > t) {
            s += 1;
        }
        
        bool c = false;
        
        //
        for (int k = 2; k < d; k++) {
            //
            if (y_n[k] > t) {
                s += 1;
            }
            
            //
            if (y_n[k] > y_n[i]) {
                j = i; 
                i = k; 
                
                c = true;
            }
            else if (y_n[k] > y_n[j]) {
                j = k;
                
                c = true;
            }
            else {
                c = true;
            }
            
            //
            if (c) {
                m = y_n[i] - y_n[j];
            }
        }
        
        // 
        if (s < 1) {
            yc[n] = "NO VALUE ABOVE THRESHOLD";
        }
        else if (m < b) {
            yc[n] = "MARGIN SMALLER THAN BUFFER";
        }
        else {
            yc[n] = C[i];
        }
    }
    
    return yc;
}