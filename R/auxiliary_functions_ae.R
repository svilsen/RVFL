f <- function(W, H, X) { 
    Xhat <- H %*% W
    return(0.5 * norm(Xhat - X, "F")^2 )
}

grad_f <- function(W, H, X){ 
    Xhat <- H %*% W
    return(t(H) %*% (Xhat - X))
}

g <- function(W) { 
    return(norm(as.matrix(W), '1'))
}

relu <- function(x) { 
    return(max(x, 0))
}

prox_g <- function(W, tau) { 
    W_tau <- abs(W) - tau
    W_tau[W_tau < 0] <- 0
    return(sign(W) * W_tau)
}

lasso_ls <- function(H, X, tau = 1, max_iterations = 1000, w = 10, step_shrink = 0.001, backtrack = TRUE, tolerance = 1e-12) {
    tau_values <- rep(NA, max_iterations)
    f_values <- rep(NA, max_iterations)
    objective <- rep(NA, max_iterations + 1)
    
    backtrack_total <- 0
    backtrack_count <- 0
    
    W_new <- matrix(runif(n = ncol(H) * ncol(X)), nrow = ncol(H), ncol = ncol(X))
    D_new <- W_new
    
    f_values[1] <- f(D_new, H, X)
    grad_f_new <- grad_f(D_new, H, X)
    
    min_objective_value <- Inf
    objective[1] <- f_values[1] + g(W_new)
    for (i in seq_len(max_iterations)) {
        W_old <- W_new
        grad_f_old <- grad_f_new
        tau_i <- tau
        
        W_new_hat <- W_old - tau_i * grad_f_old
        W_new <- prox_g(W_new_hat, tau_i)
        delta_W <- W_new - W_old
        
        D_new <- W_new
        f_new <- f(D_new, H, X)
        
        if (backtrack) {
            M <- max(f_values[max(i - w, 1):max(i - 1, 1)])
            backtrack_count <- 0
            continue <- any((f_new - 1e-12) > (M + t(delta_W) %*% grad_f_old + 0.5 * (norm(delta_W, "f")^2) / tau_i)) && (backtrack_count < 20)
            while (continue) {
                tau_i <- tau_i * step_shrink
                
                W_new_hat <- W_old - tau_i * c(grad_f_old)
                W_new <- prox_g(W_new_hat, tau_i)
                
                D_new <- W_new
                f_new <- f(D_new, H, X)
                delta_W <- W_new - W_old
                
                dWGf <- t(delta_W) %*% grad_f_old
                backtrack_count <- backtrack_count + 1
                continue <- any((f_new - 1e-12) > (M + t(delta_W) %*% grad_f_old + 0.5 * (norm(delta_W, "f")^2) / tau_i)) && (backtrack_count < 20)
            }
            
            backtrack_total <- backtrack_total + backtrack_count
        }
        
        tau_values[i] <- tau_i
        f_values[i] <- f_new
        objective[i + 1] <- f_new + g(W_new)
        
        if (objective[i + 1] < min_objective_value) {
            W_best <- W_new
            min_objective_value <- min(min_objective_value, objective[i + 1])
        }
        
        grad_f_new <- grad_f(D_new, H, X)
        D_g <- grad_f_new + (W_new_hat - W_old) / tau_i
        inner_prod <- t(c(delta_W)) %*% c(D_g)
        tau_s <- max(c(norm(delta_W, "f")^2 / inner_prod))
        tau_m <- inner_prod / norm(D_g, "f")^2
        tau_m <- max(tau_m, 0)
        
        if (norm(inner_prod) < tolerance) 
            break
        
        if ((2 * tau_m) > tau_s) {
            tau <- tau_m
        } else {
            tau <- tau_s - 0.5 * tau_m
        }
        
        if ((tau <= 0) || is.infinite(tau) || is.nan(tau)) {
            tau <- 1.5 * tau_i
        }
    } 
    
    res <- list(W = W_best, OV = min_objective_value, FV = f_values[i], Backtracking = backtrack_total)
    return(res)
}
