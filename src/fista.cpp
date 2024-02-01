#include <RcppArmadillo.h>
#include <auxiliary_functions.hpp>

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(cpp11)]]

double f(const arma::mat & X, const arma::mat & H, const arma::mat & B) {
    //
    const arma::mat X_hat = H * B;
    
    //
    const arma::mat R = X - X_hat;
    const arma::mat R_sq = R % R;
    
    //
    const double SSR = 0.5 * arma::accu(R_sq);
    return SSR;
}

arma::mat f_grad(const arma::mat & X, const arma::mat & H, const arma::mat & B) {
    //
    const arma::mat X_hat = H * B;
    const arma::mat R = X_hat - X;
    
    //
    const arma::mat H_T = H.t();
    const arma::mat G = H_T * R;
    
    //
    return G;
}

double L1(const arma::mat & B) {
    int N = B.n_rows;
    int M = B.n_cols;
    
    double max = 0.0;
    for (int m = 0; m < M; m++) {
        double max_c = 0.0;
        for (int n = 0; n < N; n++) {
            max_c += std::abs(B(n, m));
        }

        if (max < max_c) {
            max = max_c;
        }
    }
    
    return max;
}

arma::mat L1_proxy(const arma::mat & B, const double & tau) {
    const arma::mat B_abs = arma::abs(B);
    const arma::mat B_sign = matrix_sign(B);
    
    const arma::mat M_tau = tau * arma::ones(arma::size(B));
    const arma::mat B_tau = matrix_nonzero(B_abs - M_tau);
    
    const arma::mat BS = B_sign % B_tau;
    
    return BS;
}

//[[Rcpp::export()]]
Rcpp::List fista(const arma::mat & X, const arma::mat & H, const arma::mat & W, const double & tau, const int & max_iterations, 
                 const int & w, const double & step_shrink, const int & backtrack, const double & tolerance, const int & trace) {
    //
    double lambda = tau;
    
    arma::mat W_new = W;
    arma::mat W_best = W;
    
    //
    arma::colvec f_values(w, arma::fill::zeros);
    f_values[0] = f(X, H, W_new);
    
    //
    arma::mat grad_f = f_grad(X, H, W_new);
    
    //  
    double objective_new = f_values[0] + L1(W_new);

    //
    double min_objective_value = objective_new;
    int backtrack_total = 0;
    int backtrack_count = 0;
    
    //
    int k = 0;
    bool converged = false;
    while (!converged) {
        double lambda_k = lambda;
        
        //
        arma::mat W_old = W_new;

        //
        arma::mat W_new_hat = W_old - lambda_k * grad_f;
        W_new = L1_proxy(W_new_hat, lambda_k);
        arma::mat delta_W = W_new - W_old;
        
        double f_new = f(X, H, W_new);
        
        // Backtracking
        if (backtrack > 0) {
            // Finding largest f_value within window
            double M = arma::max(f_values); 
            
            // Backtracking
            arma::mat bt_bound = M + delta_W.t() * grad_f + 0.5 * arma::accu(delta_W % delta_W) / lambda_k;
            
            backtrack_count = 0;
            bool bt_continue = matrix_condition(bt_bound, f_new - tolerance) & (backtrack_count < backtrack);
            while (bt_continue) {
                //
                lambda_k = lambda_k * step_shrink;
                
                //
                W_new_hat = W_old - lambda_k * grad_f;
                W_new = L1_proxy(W_new_hat, lambda_k);
                
                //
                f_new = f(X, H, W_new);
                delta_W = W_new - W_old;
                
                //
                bt_bound = M + delta_W.t() * grad_f + 0.5 * arma::accu(delta_W % delta_W) / lambda_k;
                
                //
                backtrack_count = backtrack_count + 1;
                bt_continue = matrix_condition(bt_bound, f_new - tolerance) & (backtrack_count < backtrack);
            }
            
            backtrack_total = backtrack_total + backtrack_count;
        }
        
        // 
        int kw = k % w;
        f_values[kw] = f_new;
        objective_new = f_new + L1(W_new);
        
        if (objective_new < min_objective_value) {
            W_best = W_new;
            min_objective_value = objective_new;
        }
        
        // 
        grad_f = f_grad(X, H, W_new);
        arma::mat D_g = grad_f + (W_new_hat - W_old) / lambda_k;
        
        double inner_prod = arma::accu(delta_W % D_g);
        if ((std::abs(inner_prod) < tolerance) | (k > max_iterations)) {
            converged = true;
        }
        
        // Correcting 'lambda'
        double delta_W_prod_sum = arma::accu(delta_W % delta_W);
        double D_g_prod_sum = arma::accu(D_g % D_g);
        
        double lambda_s = delta_W_prod_sum / inner_prod;
        double lambda_m = std::max(inner_prod / D_g_prod_sum, 0.0);
        
        if ((2.0 * lambda_m) > lambda_s) {
            lambda = lambda_m;
        } else {
            lambda = lambda_s - 0.5 * lambda_m;
        }

        if ((lambda <= 0) | std::isinf(lambda) | std::isnan(lambda)) {
            lambda = 1.5 * lambda_k;
        }
        
        // 
        if (trace > 0) {
            if ((k == 1) | (k == max_iterations) | ((k % trace) == 0) | converged) {
                Rcpp::Rcout << "Iteration: " << k << " / " << max_iterations << "\n"
                            << "\tObjective: " << objective_new << "\n" 
                            << "\tInner-product: " << inner_prod << "\n" 
                            << "\tBacktrack: " << backtrack_total << "\n"
                            << "\tConverged: " << converged << "\n";
            }
        }
        
        //
        k = k + 1;
    }
    
    return Rcpp::List::create(
        Rcpp::Named("W") = W_best,
        Rcpp::Named("OV") = min_objective_value, 
        Rcpp::Named("FV") = f_values, 
        Rcpp::Named("Backtracking") = backtrack_total
    );
}
