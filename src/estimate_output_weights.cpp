#include "RcppArmadillo.h"

//[[Rcpp::depends(RcppArmadillo)]]

double rho(const arma::mat & O, const arma::colvec & y, const arma::colvec & beta, const int & j) {
    arma::colvec beta_j = beta;
    beta_j[j] = 0.0; 
    
    const arma::colvec residual = (y - O * beta_j);
    const double z = arma::as_scalar(arma::trans(O.col(j)) * residual);
    return z;
}

arma::colvec coordinate_descent(const arma::mat & O, const arma::colvec & y, 
                                const double & lambda, const arma::colvec & beta0, 
                                const int & N, const int & p) {
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
        arma::mat beta0 = Op * y;  
        beta = beta0;
        
        for (int i = 0; i < d; i++) {
            const arma::colvec & y_i = y.col(i);
            const arma::colvec & beta0_i = beta0.col(i);
            
            arma::colvec beta_i = coordinate_descent(O, y_i, lambda, beta0_i, N, p);
            beta.col(i) = beta_i;            
        }
    }
    
    const arma::mat residual = y - O * beta; 
    const arma::mat sigma_squared = arma::trans(residual) * residual / (N - p);
    return Rcpp::List::create(
        Rcpp::Named("beta") = beta,
        Rcpp::Named("sigma") = sigma_squared
    );
}
