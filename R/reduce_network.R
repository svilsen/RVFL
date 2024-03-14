#####################################################################################
####################### Reduce RWNN and ERWNN neural networks #######################
#####################################################################################

#### Reducing output-layer
##
reduce_network_output <- function(object, p, tolerance) {
    # 
    if (is.null(tolerance)) {
        tolerance <- 1e-8
    }
    
    #
    p <- ncol(X)
    
    if (any(abs(object$Weights$Output) < tolerance)) {
        # Identifying zeroes
        zero_index <- which(abs(object$Weights$Output) < tolerance)
        
        # Removing output bias
        if (object$Bias$Output) {
            if (zero_index[1] == 1) {
                object$Bias$Output <- FALSE
                object$Weights$Output <- object$Weights$Output[-1, , drop = FALSE]
                zero_index <- zero_index[-1] - 1    
            }
        }
        
        # Silencing input features 
        k <- as.numeric(object$Bias$Output)
        if (object$Combined$Input) {
            k <- k + p
            if (any(zero_index <= k)) {
                object$Weights$Hidden[[1]][zero_index[zero_index <= k] - as.numeric(object$Bias$Output) + as.numeric(object$Bias$Hidden[1]), ] <- 0
            }
            
            zero_index <- zero_index[!(zero_index <= k)]
        } 
        
        
        # Removing weights from hidden layers
        removal_index <- zero_index
        W <- length(object$Weights$Hidden)
        for (w in seq_len(W)) {
            k <- k + object$N_hidden[w]
            removal_index_w <- zero_index[zero_index <= k] - (k - object$N_hidden[w])
            object$Weights$Hidden[[w]] <- object$Weights$Hidden[[w]][, -removal_index_w, drop = FALSE]
            
            if (w < W) {
                object$Weights$Hidden[[w + 1]] <- object$Weights$Hidden[[w + 1]][-(removal_index_w + object$Bias$Hidden[w]), , drop = FALSE]
            }
            
            object$N_hidden[w] <- ncol(object$Weights$Hidden[[w]])
            zero_index <- zero_index[!(zero_index <= k)]
        }
        
        object$Weights$Output <- object$Weights$Output[-removal_index, , drop = FALSE]
    }
    
    return(object)
}

reduce_network_output_recursive <- function(object, X, tolerance) {
    # if (object$Combined$Hidden & (length(object$N_hidden) > 1))  {
    #     stop("Setting 'method' to 'output' does not work when the output of each hidden-layer is used to predict the target. ")
    # }
    
    #
    converged <- FALSE
    size_output <- dim(object$Weights$Output)[1]
    while (!converged) {
        object <- reduce_network_output(object, p, tolerance)
        converged <- size_output == dim(object$Weights$Output)[1]
        size_output <- dim(object$Weights$Output)[1]
    }
    
    return(object)
}

#### Reducing the number of weights
##
reduce_network_global <- function(object, p) {
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    weights <- quantile(abs(unlist(object$Weights)), probs = p)
    for(i in seq_along(object$Weights$Hidden)) {
        object$Weights$Hidden[[i]][abs(object$Weights$Hidden[[i]]) <= weights] <- 0
    }
    
    object$Weights$Output[abs(object$Weights$Output) <= weights] <- 0
    
    return(object)
}

##
reduce_network_uniform <- function(object, p) {
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    #  
    for(i in seq_along(object$Weights$Hidden)) {
        weights_i <- quantile(abs(unlist(object$Weights$Hidden[[i]])), probs = p)
        object$Weights$Hidden[[i]][abs(object$Weights$Hidden[[i]]) <= weights_i] <- 0
    }
    
    weights_o <- quantile(abs(unlist(object$Weights$Output)), probs = p)
    object$Weights$Output[abs(object$Weights$Output) <= weights_o] <- 0
    
    #
    return(object)
}

##
reduce_network_lamp <- function(object, p) {
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    # 
    lamp <- vector("list", length(object$Weights$Hidden) + 1)
    for(i in seq_along(lamp)[-length(lamp)]) {
        w_i <- object$Weights$Hidden[[i]]
        w_i_sq <- w_i^2
        o_i <- order(w_i_sq)
        w_i_sq <- w_i_sq[o_i] / rev(cumsum(rev(w_i_sq[o_i])))
        w_i[o_i] <- w_i_sq
        lamp[[i]] <- w_i
    }
    
    w_o <- object$Weights$Output
    w_o_sq <- w_o^2
    o_o <- order(w_o_sq)
    w_o_sq <- w_o_sq[o_o] / rev(cumsum(rev(w_o_sq[o_o])))
    w_o[o_o] <- w_o_sq
    lamp[[length(lamp)]] <- w_o
    
    # 
    weights <- quantile(abs(unlist(lamp)), probs = p)
    for(i in seq_along(object$Weights$Hidden)) {
        object$Weights$Hidden[[i]][abs(lamp[[i]]) <= weights] <- 0
    }
    
    object$Weights$Output[abs(lamp[[length(lamp)]]) <= weights] <- 0
    
    #
    return(object)
}

#### Reducing the number of neurons
##
reduce_network_apoz <- function(object, p, tolerance, X, type) {
    #
    if (is.null(type)) {
        type <- "uniform"
    }
    else {
        type <- tolower(type)
    }
    
    if (!(type %in% c("glbl", "global", "unif", "uniform"))) {
        stop("'type' should be either 'global' or 'uniform'.")
    }
    
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    #
    if (is.null(tolerance) | !is.numeric(tolerance)) {
        warning("'tolerance' is set to '1e-8' as it was either 'NULL', or not 'numeric'.")
        tolerance <- 1e-8
    }
    else if (tolerance < 0) {
        warning("'tolerance' is set to '1e-8', because it was found to be smaller than '0'.")
        tolerance <- 1e-8
    }
    
    #
    H <- rwnn_forward(X, object$Weights$Hidden, object$activation, object$Bias$Hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = object$N_hidden[i]))
    
    Z <- lapply(seq_along(H), function(i) (H[[i]] - mean(H[[i]])) / sd(H[[i]]))
    APOZ <- lapply(seq_along(Z), function(i) apply(abs(Z[[i]]) < tolerance, 2, mean))
    
    #
    if (type %in% c("glbl", "global")) {
        K <- sapply(APOZ, length)
        K_s <- cumsum(c(0, K)) + 1
        
        gobal_removal <- order(unlist(APOZ))[-seq_len(round((1 - p) * (K_s[length(K_s)] - 1)))]
        remove_cols <- lapply(seq_along(K_s[-length(K_s)]), function(i) gobal_removal[which((gobal_removal >= K_s[i]) & (gobal_removal < K_s[i + 1]))] - (K_s[i] - 1))
    }
    else if (type %in% c("unif", "uniform")) {
        remove_cols <- lapply(APOZ, function(x) order(x)[-seq_len(round((1 - p) * length(x)))])
    }
    
    #
    W <- length(object$Weights$Hidden)
    for (w in seq_len(W)) {
        ##
        remove_cols_w <- remove_cols[[w]]
        object$Weights$Hidden[[w]] <- object$Weights$Hidden[[w]][, -remove_cols_w, drop = FALSE]
        object$N_hidden[w] <- ncol(object$Weights$Hidden[[w]])
        
        ##
        if (w < W) {    
            remove_rows_w <- remove_cols_w + as.numeric(object$Bias$Hidden[w + 1])
            object$Weights$Hidden[[w + 1]] <- object$Weights$Hidden[[w + 1]][-remove_rows_w, , drop = FALSE]
        }
        
        if (object$Combined$Hidden | (w == W)) {
            index_offset <- object$Bias$Output + p * object$Combined$Input
            if (w > 1) {
                previous_w <- sapply(object$Weights$Hidden[seq_len(w - 1)], function(x) dim(x)[2])
                index_offset <- index_offset + sum(previous_w)
            } 
            
            remove_rows_out_w <- remove_cols_w + index_offset
            object$Weights$Output <- object$Weights$Output[-remove_rows_out_w, , drop = FALSE]
        }
    }
    
    #
    return(object)
}

##
reduce_network_l2  <- function(object, p, X, type) {
    #
    if (is.null(type)) {
        type <- "uniform"
    }
    else {
        type <- tolower(type)
    }
    
    if (!(type %in% c("glbl", "global", "unif", "uniform"))) {
        stop("'type' should be either 'global' or 'uniform'.")
    }
    
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    }
    else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    }
    else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    #
    H <- rwnn_forward(X, object$Weights$Hidden, object$activation, object$Bias$Hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = object$N_hidden[i]))
    
    Z <- lapply(seq_along(H), function(i) (H[[i]] - mean(H[[i]])) / sd(H[[i]]))
    L <- lapply(seq_along(Z), function(i) apply(Z[[i]], 2, function(x) sqrt(sum(x^2))))
    
    #
    if (type %in% c("glbl", "global")) {
        K <- sapply(L, length)
        K_s <- cumsum(c(0, K)) + 1
        
        gobal_removal <- order(unlist(L))[-seq_len(round((1 - p) * (K_s[length(K_s)] - 1)))]
        remove_cols <- lapply(seq_along(K_s[-length(K_s)]), function(i) gobal_removal[which((gobal_removal >= K_s[i]) & (gobal_removal < K_s[i + 1]))] - (K_s[i] - 1))
    }
    else if (type %in% c("unif", "uniform")) {
        remove_cols <- lapply(L, function(x) order(x)[-seq_len(round((1 - p) * length(x)))])
    }
    
    
    #
    W <- length(object$Weights$Hidden)
    for (w in seq_len(W)) {
        ##
        remove_cols_w <- remove_cols[[w]]
        object$Weights$Hidden[[w]] <- object$Weights$Hidden[[w]][, -remove_cols_w, drop = FALSE]
        object$N_hidden[w] <- ncol(object$Weights$Hidden[[w]])
        
        ##
        if (w < W) {    
            remove_rows_w <- remove_cols_w + as.numeric(object$Bias$Hidden[w + 1])
            object$Weights$Hidden[[w + 1]] <- object$Weights$Hidden[[w + 1]][-remove_rows_w, , drop = FALSE]
        }
        
        if (object$Combined$Hidden | (w == W)) {
            index_offset <- object$Bias$Output + p * object$Combined$Input
            if (w > 1) {
                previous_w <- sapply(object$Weights$Hidden[seq_len(w - 1)], function(x) dim(x)[2])
                index_offset <- index_offset + sum(previous_w)
            } 
            
            remove_rows_out_w <- remove_cols_w + index_offset
            object$Weights$Output <- object$Weights$Output[-remove_rows_out_w, , drop = FALSE]
        }
    }
    
    #
    return(object)
}

##
reduce_network_correlation <- function(object, type, rho, X) {
    if (is.null(type)) {
        type <- "pearson"
    } else {
        type <- tolower(type)
    }
    
    if (is.null(rho) | !is.numeric(rho)) {
        warning("'rho' is set to '0.99' as it was either 'NULL', or not 'numeric'.")
        rho <- 0.99
    } else if (rho < 0) {
        warning("'rho' is set to '0', because it was found to be smaller than '0'.")
        rho <- 0.0
    } else if (rho > 1) {
        warning("'rho' is set to '1', because it was found to be larger than '1'.")
        rho <- 1.0
    }
    
    p <- dim(object$Weights$Hidden[[1]])[1] - object$Bias$Hidden[1]
    W <- length(object$Weights$Hidden)
    for (w in seq_len(W)) {
        ##
        H_w <- rwnn_forward(X, object$Weights$Hidden[seq_len(w)], object$activation[seq_len(w)], object$Bias$Hidden[seq_len(w)])
        H_w <- lapply(seq_along(H_w), function(i) matrix(H_w[[i]], ncol = object$N_hidden[i]))
        H_w <- H_w[[w]]
        
        ##
        C_w <- cor(H_w, method = type)
        C_w <- upper.tri(C_w) * C_w
        
        remove_cols_w <- which(apply(C_w >= rho, 2, any))
        
        ##
        object$Weights$Hidden[[w]] <- object$Weights$Hidden[[w]][, -remove_cols_w, drop = FALSE]
        object$N_hidden[w] <- ncol(object$Weights$Hidden[[w]])
        
        ##
        if (w < W) {    
            remove_rows_w <- remove_cols_w + as.numeric(object$Bias$Hidden[w + 1])
            object$Weights$Hidden[[w + 1]] <- object$Weights$Hidden[[w + 1]][-remove_rows_w, , drop = FALSE]
        }
        
        if (object$Combined$Hidden | (w == W)) {
            index_offset <- object$Bias$Output + p * object$Combined$Input
            if (w > 1) {
                previous_w <- sapply(object$Weights$Hidden[seq_len(w - 1)], function(x) dim(x)[2])
                index_offset <- index_offset + sum(previous_w)
            } 
            
            remove_rows_out_w <- remove_cols_w + index_offset
            object$Weights$Output <- object$Weights$Output[-remove_rows_out_w, , drop = FALSE]
        }
    }
    
    return(object)
}

##
reduce_network_correlation_ft <- function(object, type, rho, alpha, X) { 
    if (is.null(type)) {
        type <- "pearson"
    }
    else {
        type <- tolower(type)
    }
    
    if (is.null(rho) | !is.numeric(rho)) {
        warning("'rho' is set to '0.99' as it was either 'NULL', or not 'numeric'.")
        rho <- 0.99
    } else if (rho < 0) {
        warning("'rho' is set to '0', because it was found to be smaller than '0'.")
        rho <- 0.0
    } else if (rho > 1) {
        warning("'rho' is set to '1', because it was found to be larger than '1'.")
        rho <- 1.0
    }
    
    if (is.null(alpha) | !is.numeric(alpha)) {
        warning("'alpha' is set to '0.05' as it was either 'NULL', or not 'numeric'.")
        alpha <- 0.05
    } else if (alpha < 0) {
        warning("'alpha' is set to '0', because it was found to be smaller than '0'.")
        alpha <- 0.0
    } else if (alpha > 1) {
        warning("'alpha' is set to '1', because it was found to be larger than '1'.")
        alpha <- 1.0
    }
    
    p <- dim(object$Weights$Hidden[[1]])[1] - object$Bias$Hidden[1]
    N <- dim(X)[1]
    
    W <- length(object$Weights$Hidden)
    for (w in seq_len(W)) {
        ##
        H_w <- rwnn_forward(X, object$Weights$Hidden[seq_len(w)], object$activation[seq_len(w)], object$Bias$Hidden[seq_len(w)])
        H_w <- lapply(seq_along(H_w), function(i) matrix(H_w[[i]], ncol = object$N_hidden[i]))
        H_w <- H_w[[w]]
        
        ##
        C_w <- cor(H_w, method = type)
        C_w <- upper.tri(C_w) * C_w
        
        Z_w <- 0.5 * (log(1 + C_w) - log(1 - C_w))
        R_w <- 0.5 * (log(1 + rho) - log(1 - rho))
        
        ## 
        T_w <- (Z_w - R_w) * sqrt(N - 3)
        P_w <- upper.tri(T_w) * pnorm(T_w, 0, 1)
        
        ## 
        remove_cols_w <- which(apply(P_w < alpha, 2, all))
        
        ##
        object$Weights$Hidden[[w]] <- object$Weights$Hidden[[w]][, -remove_cols_w, drop = FALSE]
        object$N_hidden[w] <- ncol(object$Weights$Hidden[[w]])
        
        ##
        if (w < W) {    
            remove_rows_w <- remove_cols_w + as.numeric(object$Bias$Hidden[w + 1])
            object$Weights$Hidden[[w + 1]] <- object$Weights$Hidden[[w + 1]][-remove_rows_w, , drop = FALSE]
        }
        
        if (object$Combined$Hidden | (w == W)) {
            index_offset <- object$Bias$Output + p * object$Combined$Input
            if (w > 1) {
                previous_w <- sapply(object$Weights$Hidden[seq_len(w - 1)], function(x) dim(x)[2])
                index_offset <- index_offset + sum(previous_w)
            } 
            
            remove_rows_out_w <- remove_cols_w + index_offset
            object$Weights$Output <- object$Weights$Output[-remove_rows_out_w, , drop = FALSE]
        }
    }
    
    return(object)
}

#### 
##
reduce_network_relief <- function(object, p, X, type) {
    # 
    if (is.null(type)) {
        type <- "neuron"
    } else {
        type <- tolower(type)
    }
    
    if (!(type %in% c("w", "weight", "n", "neuron"))) {
        stop("'type' should be either 'weight' or 'neuron'.")
    }
    
    #
    if (is.null(p) | !is.numeric(p)) {
        warning("'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")
        p <- 0.1
    } else if (p < 0) {
        warning("'p' is set to '0.01', because it was found to be smaller than '0'.")
        p <- 0.01
    } else if (p > 1) {
        warning("'p' is set to '0.99', because it was found to be larger than '1'.")
        p <- 0.99
    }
    
    #
    k <- ncol(X)
    
    #
    H <- rwnn_forward(X, object$Weights$Hidden, object$activation, object$Bias$Hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = object$N_hidden[i]))
    
    #
    C <- append(list(X), H)
    W <- append(object$Weights$Hidden, list(object$Weights$Output))
    B <- c(object$Bias$Hidden, object$Bias$Output)
    
    #
    for (w in seq_along(C)) {
        # 
        if (w < length(C)) {
            C_w <- C[[w]]
        } else {
            if (object$Combined$Hidden){
                C_w <- do.call("cbind", H)
            }
            else {
                C_w <- H[[length(H)]]
            }
            
            if (object$Combined$Input) {
                C_w <- cbind(X, C_w)
            }
        }
        
        #
        W_w <- W[[w]]
        
        #
        if (B[w]) {
            b_w <- W_w[1, , drop = FALSE]
            W_w <- W_w[-1, , drop = FALSE]
        } else {
            b_w <- matrix(0, nrow = 1, ncol = ncol(W_w))
        }
        
        #
        I_w <- importance_score(C_w, W_w)
        N_w <- matrix(apply(I_w, 2, sum) + abs(b_w), nrow = 1)
        
        S_w <- I_w / matrix(N_w, nrow = nrow(I_w), ncol = ncol(I_w), byrow = TRUE)
        
        if (B[w]) {
            B_w <- abs(b_w) / N_w
            S_w <- rbind(B_w, S_w)
        }
        
        # 
        if (type %in% c("w", "weight")) {
            R_w <- quantile(S_w, probs = p)
            if (w < length(C)) {
                object$Weights$Hidden[[w]][S_w <= R_w] <- 0
            } 
            else {
                object$Weights$Output[S_w <= R_w] <- 0
            }
        }
        else if (type %in% c("n", "neuron")) {
            #
            if (w == length(C)) {
                next
            }
            
            N_w <- N_w / sum(N_w) 
            R_w <- quantile(N_w, probs = p)
            
            remove_cols_w <- which(N_w < R_w)
            object$Weights$Hidden[[w]] <- object$Weights$Hidden[[w]][, -remove_cols_w, drop = FALSE]
            object$N_hidden[w] <- ncol(object$Weights$Hidden[[w]])
            
            #
            if (w < (length(C) - 1)) {    
                remove_rows_w <- remove_cols_w + as.numeric(object$Bias$Hidden[w + 1])
                object$Weights$Hidden[[w + 1]] <- object$Weights$Hidden[[w + 1]][-remove_rows_w, , drop = FALSE]
            }
            
            if (object$Combined$Hidden | (w == (length(C) - 1))) {
                index_offset <- object$Bias$Output + k * object$Combined$Input
                if (w > 1) {
                    previous_w <- sapply(object$Weights$Hidden[seq_len(w - 1)], function(x) dim(x)[2])
                    index_offset <- index_offset + sum(previous_w)
                } 
                
                remove_rows_out_w <- remove_cols_w + index_offset
                object$Weights$Output <- object$Weights$Output[-remove_rows_out_w, , drop = FALSE]
            }
        }
    }
    
    #
    return(object)
}



####
#' @title Reduce the weights of a random weight neural network.
#' 
#' @description Methods for weight and neuron pruning in random weight neural networks.
#' 
#' @param object An \link{RWNN} or \link{ERWNN}-object.
#' @param method A string setting the method, or a function, used to reduce the network  (see details).
#' @param retrain TRUE/FALSE: Should the output weights be retrained after reduction?
#' @param ... Additional arguments passed to the reduction method (see details).
#' 
#' @details ... 
#' 
#' @return A reduced \link{RWNN} or \link{ERWNN}-object.
#' 
#' @export
reduce_network <- function(object, method, retrain = TRUE, ...) {
    UseMethod("reduce_network")
}

#' @rdname reduce_network
#' @method reduce_network RWNN
#'
#' @example inst/examples/reduction_example.R
#'
#' @export
reduce_network.RWNN <- function(object, method, retrain = TRUE, ...) {
    ##
    dots <- list(...)
    
    ##
    if (is.null(retrain) | !is.logical(retrain)) {
        warning("'retrain' is set to 'TRUE' as it was either 'NULL', or not 'logical'.")
        retrain <- TRUE
    }
    
    ##
    if ((!is.null(dots$X)) & (!is.null(dots$y))) {
        X <- dots$X
        y <- dots$y
    } else if (!is.null(object$data$X)) {
        X <- object$data$X
        y <- object$data$y
    } else {
        stop("Data has to be present in the model object, or supplied through '...' argument as 'X = ' and 'y = '.")
    }
    
    ##
    if (method %in% c("mag", "magnitide", "glbl", "global")) {
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (globally).
        object <- reduce_network_global(object = object, p = dots$p)
    }
    else if (method %in% c("unif", "uniform")) {
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (uniformly layer-by-layer).
        object <- reduce_network_uniform(object = object, p = dots$p)
    }
    else if (method %in% c("lamp")) {
        ## Weight pruning method: Reducing the number of hidden-weights using the LAMP scores.
        object <- reduce_network_lamp(object = object, p = dots$p)
    }
    else if (method %in% c("apoz")) {
        ## Neuron pruning method: Average percentage of "zeros".
        if (!all(object$activation == "relu")) {
            warning("APOZ was designed for 'relu' activation functions, but no 'relu' activation was found.")
        }
        
        object <- reduce_network_apoz(object = object, p = dots$p, tolerance = dots$tolerance, X = X, type = dots$type)
    }
    else if (method %in% c("l2")) {
        ## Neuron pruning method: L2-norm of hidden-weights.
        object <- reduce_network_l2(object = object, p = dots$p, X = X, type = dots$type)
    }
    else if (method %in% c("cor", "correlation")) {
        ## Neuron pruning method: Correlation between activated neurons.
        object <- reduce_network_correlation(object = object, type = dots$type, rho = dots$rho, X = X)
    }
    else if (method %in% c("ct", "cortest", "correlationtest")) {
        ## Neuron pruning method: Correlation between activated neurons.
        object <- reduce_network_correlation_ft(object = object, type = dots$type, rho = dots$rho, alpha = dots$alpha, X = X)
    }
    else if (method %in% c("relief")) {
        ## Neuron and weight pruning method: Reduction based on relief scores.
        object <- reduce_network_relief(object = object, p = dots$p, X = X, type = dots$type)
    }
    else if (method %in% c("output")) {
        ## Removing '0' weights from the output-layer.
        object <- reduce_network_output(object = object, p = ncol(X), tolerance = dots$tolerance)
    } 
    else if (is.function(method)) {
        object_list <- list(object = object, X = X, y = y) |> append(dots)
        object <- do.call(method, object_list)
    }
    else {
        stop("'method' is either not implemented, or not a function.")
    }
    
    ## 
    for (w in seq_along(object$Weights$Hidden)) {
        if (object$Bias$Hidden[w]) {
            if (sum(abs(object$Weights$Hidden[[w]][1, ])) < 1e-8) {
                object$Weights$Hidden[[w]] <- object$Weights$Hidden[[w]][-1, , drop = FALSE]
                object$Bias$Hidden[w] <- FALSE
            }
        }
    }
    
    if (object$Bias$Output) { 
        if (abs(object$Weights$Output[1]) < 1e-8) {
            object$Weights$Output <- object$Weights$Output[-1, , drop = FALSE]
            object$Bias$Output <- FALSE
        }
    }
    
    ## 
    if (retrain) {
        H <- rwnn_forward(X, object$Weights$Hidden, object$activation, object$Bias$Hidden)
        H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = object$N_hidden[i]))
        
        if (object$Combined$Hidden){
            H <- do.call("cbind", H)
        } else {
            H <- H[[length(H)]]
        }
        
        O <- H
        if (object$Combined$Input) {
            O <- cbind(X, H)
        }
        
        if (object$Bias$Output) {
            O <- cbind(1, O)
        }
        
        W <- estimate_output_weights(O, y, object$lnorm[length(object$lnorm)], object$lambda[length(object$lambda)])
        object$Weights$Output <- W$beta
        object$Sigma$Output <- W$sigma
    }
    
    ##
    return(object)
}

#' @rdname reduce_network
#' @method reduce_network ERWNN
#' 
#' @export
reduce_network.ERWNN <- function(object, method, retrain = TRUE, ...) {
    dots <- list(...)
    
    if ((!is.null(dots$X)) & (!is.null(dots$y))) {
        X <- dots$X
        y <- dots$y
    } else if (!is.null(object$data$X)) {
        X <- object$data$X
        y <- object$data$y
    } else {
        stop("Data has to be present in the model object, or supplied through '...' argument as 'X = ' and 'y = '.")
    }
    
    B <- length(object$RWNNmodels)
    for (b in seq_len(B)) {
        list_b <- list(object = object$RWNNmodels[[b]], method = method, retrain = retrain, X = X, y = y) |> append(dots)
        object_b <- do.call(reduce_network, list_b)
        
        object$RWNNmodels[[b]] <- object_b
    }
    
    return(object)
}

