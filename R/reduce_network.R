#####################################################################################
####################### Reduce RWNN and ERWNN neural networks #######################
#####################################################################################

#### Reducing output-layer
##
reduce_network_output <- function(object) {
    if (any(abs(object$Weights$Output) < 1e-8)) {
        #
        zero_index <- which(abs(object$Weights$Output) < 1e-8)
        output_dim <- ncol(object$Weights$Output)
        
        # Including off-set in case a functional link is included
        k <- 0
        if (object$Combined$Input) {
            k <- dim(object$Weights$Hidden[[1]])[1] - object$Bias$Hidden[1]
        }
        
        # Removing output bias
        if ((object$Bias$Output) & ((k + 1) %in% zero_index)) {
            object$Bias$Output <- FALSE
            object$Weights$Output <- as.matrix(object$Weights$Output[-(k + 1), ], ncol = output_dim)
            zero_index <- zero_index[-1] - 1    
        }
        
        # Removing weights with zero output
        last_layer <- seq_len(object$N_hidden[length(object$N_hidden)]) + length(object$Weights$Output) - object$N_hidden[length(object$N_hidden)]
        zero_index <- zero_index[zero_index %in% last_layer]
        if (length(zero_index) > 0) {
            #
            object$Weights$Output <- as.matrix(object$Weights$Output[-zero_index, ], ncol = output_dim)
            
            #
            object$N_hidden[length(object$N_hidden)] <- object$N_hidden[length(object$N_hidden)] - sum(last_layer %in% zero_index)
            
            #
            if (object$N_hidden[length(object$N_hidden)] == 0) {
                object$N_hidden <- object$N_hidden[-length(object$N_hidden)]
                object$Weights$Hidden <- object$Weights$Hidden[[-length(object$Weights$Hidden)]]
            } 
            else {
                object$Weights$Hidden[[length(object$Weights$Hidden)]] <- as.matrix(object$Weights$Hidden[[length(object$Weights$Hidden)]][, !last_layer %in% zero_index], ncol = object$N_hidden[length(object$N_hidden)])
            }
        }
    }
    
    return(object)
}

reduce_network_output_recursive <- function(object) {
    if (object$Combined$Hidden)  {
        stop("Setting 'method' to 'last' does not work when the output of each hidden-layer is used to predict the target. ")
    }
    
    p <- length(object$Weights$Output)
    N <- dim(X)[1]
    
    converged <- FALSE
    size_output <- dim(object$Weights$Output)[1]
    while (!converged) {
        object <- reduce_network_output(object)
        converged <- size_output == dim(object$Weights$Output)[1]
        size_output <- dim(object$Weights$Output)[1]
    }
    
    object$Sigma$Output <- ((N - p) / (N - length(object$Weights$Output))) * object$Sigma$Output
    
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
    
    # Create LAMP scores
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
    
    # Reduce based on LAMP scores
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
reduce_network_correlation <- function(object, type, rho, X) {
    if (is.null(type)) {
        type <- "pearson"
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
        
        remove_cols_w <- which(apply(C_w >= rho, 1, any))
        
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

#' @title Reduce the weights of a random weight neural network.
#' 
#' @description Methods for weight and neuron pruning in random weight neural networks.
#' 
#' @param object An \link{RWNN} or \link{ERWNN}-object.
#' @param method A string setting the method (see details), or a function, used to reduce the network.
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
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (uniformly by layer).
        object <- reduce_network_uniform(object = object, p = dots$p)
    }
    else if (method %in% c("lamp")) {
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (globally using the LAMP score).
        object <- reduce_network_lamp(object = object, p = dots$p)
    }
    else if (method %in% c("last")) {
        ## Weight pruning method: Removing '0' weights from the output-layer.
        object <- reduce_network_output_recursive(object)
    } 
    else if (method %in% c("apoz")) {
        ## Neuron pruning method: Average percentage of "zeros".
        
    }
    else if (method %in% c("l2")) {
        ## Neuron pruning method: L2-norm of hidden-weights.
        
    }
    else if (method %in% c("cor", "correlation")) {
        ## Neuron pruning method: Correlation between activated neurons.
        object <- reduce_network_correlation(object = object, type = dots$type, rho = dots$rho, X = X)
    }
    else if (is.function(method)) {
        object_list <- list(object = object, X = X, y = y) |> append(dots)
        object <- do.call(method, object_list)
    }
    else {
        stop("'method' is either not implemented, or not a function.")
    }
    
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
reduce_network.ERWNN <- function(object, method = NULL, retrain = TRUE, ...) {
    dots <- list(...)
    
    if ((!is.null(dots$X)) & (!is.null(dots$y))) {
        X <- dots$X
        y <- dots$y
    }
    else if (!is.null(object$data$X)) {
        X <- object$data$X
        y <- object$data$y
    }
    else {
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

