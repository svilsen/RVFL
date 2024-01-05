#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

double rss(const arma::mat & O, const arma::mat & y, const double & lambda, const arma::mat & beta) {
    const arma::mat residual = y - O * beta;
    const double rss = arma::as_scalar(arma::trans(residual) * residual);
    const double l1 = arma::accu(arma::abs(beta));
    return rss + lambda * l1;
}

double rho(const arma::mat & O, const arma::mat & y, const arma::mat & beta, const int & j, const int & k) {
    arma::mat beta_jk = beta;
    beta_jk(j, k) = 0.0; 
    
    const arma::mat residual = (y - O * beta_jk);
    const double z = arma::as_scalar(arma::accu(arma::trans(O.col(j)) * residual));
    return z;
}

arma::mat coordinate_descent(const arma::mat & O, const arma::mat & y, 
                                const double & lambda, const arma::mat & beta0, 
                                const int & N, const int & p, const int & d) {
    arma::mat z(p, d, arma::fill::ones);
    for (int n = 0; n < N; n++) {
        const arma::mat o = arma::trans(O.row(n));
        z += o % o;
    }
    
    int i = 0;
    bool converged = false;
    double e_old, e_new = HUGE_VAL;
    arma::mat beta = beta0;
    while (!converged) {
        e_old = e_new;
        e_new = 0.0;
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < d; k++) {
                double beta_jk = beta(j, k);
                double rho_jk = rho(O, y, beta, j, k);
                if (rho_jk < (-lambda / 2.0)) {
                    beta(j, k) = (rho_jk + lambda / 2.0) / z(j, k);
                }
                else if (rho_jk > (lambda / 2.0)) {
                    beta(j, k) = (rho_jk - lambda / 2.0) / z(j, k);
                }
                else {
                    beta(j, k) = 0.0;
                }
                
                if (std::abs(beta[j] - beta_jk) > e_new) {
                    e_new = std::abs(beta[j] - beta_jk);
                }
            }
        }
        
        converged = (std::abs(e_old - e_new) < 1e-8) || (i > 1000);
        i++;
    }
    
    return beta;
}

//[[Rcpp::export]]
Rcpp::List estimate_output_weights(const arma::mat & O, const arma::mat & y, const std::string & lnorm, const double & lambda) {
    const int & N = O.n_rows;
    const int & p = O.n_cols;
    const int & d = y.n_cols;
    
    arma::mat beta;
    if (lambda < 1e-8) {
        arma::mat Op;
        if (p <= N) {
            arma::mat Q, R;
            arma::qr_econ(Q, R, O);
            
            const arma::mat & Ri = arma::inv(R);
            const arma::mat & QT = arma::trans(Q);
            
            Op = Ri * QT; 
        }
        else {
            Op = arma::pinv(O); 
        }
        
        beta = Op * y;
    }
    else if (lnorm == "l2") {
        arma::mat Op;
        const arma::mat & OT = arma::trans(O);
        
        if (p <= N) {
            const arma::mat & Ip = lambda * arma::eye(p, p);
            const arma::mat & OTO = OT * O;
            const arma::mat & Oi = arma::inv(OTO + Ip);
            Op = Oi * OT;
        }
        else {
            const arma::mat & IN = lambda * arma::eye(N, N);
            const arma::mat & OOT = O * OT;
            const arma::mat & Oi = arma::inv(OOT + IN);
            Op = OT * Oi;
        }
        
        beta = Op * y;  
    }
    else {
        const arma::mat Op = arma::pinv(O); 
        const arma::mat beta0 = Op * y;  
        
        beta = coordinate_descent(O, y, lambda, beta0, N, p, d);  
    }
    
    const arma::mat residual = y - O * beta; 
    const double sigma_squared = arma::as_scalar(arma::accu(arma::trans(residual) * residual / (N - p)));
    return Rcpp::List::create(
        Rcpp::Named("beta") = beta,
        Rcpp::Named("sigma") = std::sqrt(sigma_squared)
    );
}

