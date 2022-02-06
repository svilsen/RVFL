#######################################################################################
####################### Ensemble Deep ERWNN neural network ############################
#######################################################################################

#' @title Ensemble deep random weight neural networks
#' 
#' @description Use multiple layers to create deep ensemble random weight neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link{tibble}) used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of each RWNN.
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{ERWNN-object}.
#' 
#' @export
ed_rwnn <- function(X, y, formula, data, N_hidden, lambda = 0, control = list()) {
    UseMethod("ed_rwnn")
}

#' @rdname ed_rwnn
#' @method ed_rwnn default
#' 
#' @example inst/examples/edrwnn_example.R
#' 
#' @export
ed_rwnn.default <- function(X, y, N_hidden, lambda = 0, control = list()) {
    ## Checks
    control$N_hidden <- N_hidden
    control <- do.call(control_rwnn, control)
    lnorm <- control$lnorm
    control$lnorm <- "l2"
    
    dc <- data_checks(y, X)
    
    ##
    deeprwnn <- rwnn(X = X, y = y, N_hidden = N_hidden, lambda = lambda, control = control)
    H <- rwnn_forward(X = X, W = deeprwnn$Weights$Hidden, activation = deeprwnn$activation, bias = deeprwnn$Bias$Hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = deeprwnn$N_hidden[i]))
    O <- lapply(seq_along(H), function(i) cbind(1, X, H[[i]]))
    beta <- lapply(seq_along(O), function(i) estimate_output_weights(O[[i]], y, lnorm, lambda)$beta)
    
    ##
    object <- list(
        data = list(X = X, y = y), 
        RWNNmodels = deeprwnn, 
        OutputWeights = beta,
        weights = rep(1L / length(N_hidden), length(N_hidden)), 
        method = "ed"
    ) 
    
    class(object) <- "ERWNN"
    return(object)
}


#' @rdname ed_rwnn
#' @method ed_rwnn formula
#' 
#' @example inst/examples/edrwnn_example.R
#' 
#' @export
ed_rwnn.formula <- function(formula, data, N_hidden, lambda = 0, control = list()) {
    if (missing(formula)) {
        stop("'formula' needs to be supplied when using 'data'.")
    }
    
    if (missing(data)) {
        stop("'data' needs to be supplied when using 'formula'.")
    }
    
    #
    X <- model.matrix(formula, data)
    keep <- which(colnames(X) != "(Intercept)")
    if (any(colnames(X) == "(Intercept)")) {
        X <- X[, keep]
    }
    
    X <- as.matrix(X, ncol = length(keep))
    
    #
    y <- as.matrix(model.response(model.frame(formula, data)), nrow = nrow(data))
    
    #
    mm <- ed_rwnn(X, y, N_hidden = N_hidden, lambda = lambda, control = control)
    return(mm)
}