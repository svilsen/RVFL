#####################################################################################
####################### Reduce RWNN and ERWNN neural networks #######################
#####################################################################################

##
reduce_network_output <- function(object) {
    if (any(abs(object$Weights$Output) < 1e-8)) {
        #
        zero_index <- which(abs(object$Weights$Output) < 1e-8)
        output_dim <- ncol(object$Weights$Output)
        
        # Including off-set in case the a functional link is included
        k <- 0
        if (object$Combined) {
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

#' @title Reduce the weights of a random weight neural network.
#' 
#' @description Methods for weight and neuron pruning in random weight neural networks.
#' 
#' @param object An \link{RWNN} or \link{ERWNN}-object.
#' @param method A string setting the method used to reduce the network (see details).
#' @param ... Additional arguments passed to the reduction method (see details).
#' 
#' @details ... 
#' 
#' @return A reduced \link{RWNN} or \link{ERWNN}-object.
#' 
#' @export
reduce_network <- function(object, method = NULL, ...) {
    UseMethod("reduce_network")
}

#' @rdname reduce_network
#' @method reduce_network RWNN
#'
#' @example inst/examples/reduction_example.R
#'
#' @export
reduce_network.RWNN <- function(object, method = NULL, ...) {
    ##
    dots <- list(...)
    
    ##
    if (is.null(method)) {
        method <- "lamp"
    }
    else {
        method <- tolower(method)
    }
    
    ##
    if (!(method %in% c("last", "lasso"))) {
        p <- dots$p
        
        if (is.null(p) | !is.numeric(p)) {
            warning("'p' is set to '0.1' as it was either not supplied, or not numeric.")
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
    }
    
    if (method %in% c("mag", "magnitide", "glbl", "global")) {
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (globally).
        weights <- quantile(abs(unlist(object$Weights)), probs = p)
        for(i in seq_along(object$Weights$Hidden)) {
            object$Weights$Hidden[[i]][abs(object$Weights$Hidden[[i]]) <= weights] <- 0
        }
        
        object$Weights$Output[abs(object$Weights$Output) <= weights] <- 0
    }
    else if (method %in% c("unif", "uniform")) {
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (uniformly by layer).
        for(i in seq_along(object$Weights$Hidden)) {
            weights_i <- quantile(abs(unlist(object$Weights$Hidden[[i]])), probs = p)
            object$Weights$Hidden[[i]][abs(object$Weights$Hidden[[i]]) <= weights_i] <- 0
        }
        
        weights_o <- quantile(abs(unlist(object$Weights$Output)), probs = p)
        object$Weights$Output[abs(object$Weights$Output) <= weights_o] <- 0
    }
    else if (method %in% c("lamp")) {
        ## Weight pruning method: Reducing the number of hidden-weights based on magnitude (globally using the LAMP score).
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
    }
    else if (method %in% c("last", "lasso")) {
        ## Weight pruning method: Removing '0' weights from the output-layer.
        p <- length(object$Weights$Output)
        N <- dim(object$data$X)[1]
        
        converged <- FALSE
        size_output <- dim(object$Weights$Output)[1]
        while (!converged) {
            object <- reduce_network_output(object)
            converged <- size_output == dim(object$Weights$Output)[1]
            size_output <- dim(object$Weights$Output)[1]
        }
        
        object$Sigma$Output <- ((N - p) / (N - length(object$Weights$Output))) * object$Sigma$Output
    } 
    else if (method %in% c("apoz")) {
        ## Neuron pruning method: Average percentage of "zeros".
        
    }
    else if (method %in% c("l2")) {
        ## Neuron pruning method: L2-norm of hidden-weights.
        
    }
    else {
        stop("Provided 'method' not implemented.")
    }
    
    ##
    return(object)
}

#' @rdname reduce_network
#' @method reduce_network ERWNN
#' 
#' @export
reduce_network.ERWNN <- function(object, method = NULL, ...) {
    B <- length(object$RWNNmodels)
    for (b in seq_len(B)) {
        object$RWNNmodels[[b]] <- reduce_network(object = object$RWNNmodels[[b]], method = method, p = p, ...)
    }
    
    return(object)
}

## 

#' @title Reduce the weights of a random weight neural network.
#' 
#' @description Methods for weight and neuron pruning in random weight neural networks.
#' 
#' @param object An \link{RWNN} or \link{ERWNN}-object.
#' @param ... Arguments passed to the \link{reduce_network}-function.
#' 
#' @return A reduced \link{RWNN} or \link{ERWNN}-object.
#' 
#' @export
reduce_retrain_network <- function(object, ...) {
    UseMethod("reduce_retrain_network")
}

#' @rdname reduce_retrain_network
#' @method reduce_retrain_network RWNN
#' 
#' @export
reduce_retrain_network.RWNN <- function(object, ...) {
    
}

#' @rdname reduce_retrain_network
#' @method reduce_retrain_network ERWNN
#' 
#' @export
reduce_retrain_network.ERWNN <- function(object, ...) {
    B <- length(object$RWNNmodels)
    for (b in seq_len(B)) {
        object$RWNNmodels[[b]] <- reduce_retrain_network(object = object$RWNNmodels[[b]], ...)
    }
    
    return(object)
}

