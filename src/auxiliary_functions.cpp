#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

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