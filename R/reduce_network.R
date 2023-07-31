#####################################################################################
####################### Reduce RWNN and ERWNN neural networks #######################
#####################################################################################

reduce_network_lasso <- function(object, control) {
    if (any(abs(object$Weights$Output) < 1e-8)) {
        #
        zero_index <- which(abs(object$Weights$Output) < 1e-8)
        output_dim <- ncol(object$Weights$Output)
        
        # Including off-set in case the a functional link is included
        k <- 0
        if (control$combine_input) {
            k <- dim(object$Weights$Hidden[[1]])[1] - control$bias_hidden[1]
        }
        
        # Removing output bias
        if ((control$bias_output) & ((k + 1) %in% zero_index)) {
            control$bias_output <- FALSE
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

#' @title Reduce random weight neural network
#' 
#' @description .
#' 
#' @param object An \link{RWNN-object} or \link{ERWNN-object}.
#' @param type A string setting the method used to reduce the network.
#' 
#' @return A reduce \link{RWNN-object}.
#' 
#' @export
reduce_network <- function(object, type = "FI") {
    UseMethod("reduce_network")
}

#' @rdname reduce_network
#' @method reduce_network RWNN
#' 
#' @export
reduce_network.RWNN <- function(object, type = "FI") {
    
}

#' @rdname reduce_network
#' @method reduce_network ERWNN
#' 
#' @export
reduce_network.ERWNN <- function(object, type = "FI") {
    
}

