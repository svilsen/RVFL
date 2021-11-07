################################################################################
####################### Re-sampling ERWNN neural network #######################
################################################################################

#' @title sample_rwnn control function
#' 
#' @description A function used to create a control-object for the \link{sample_rwnn} function.
#' 
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lnorm A string indicating the regularisation norm used when estimating the weights in the output layer (either \code{"l1"} or \code{"l2"}).
#' @param bias_hidden A vector of TRUE/FALSE values. The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param activation A vector of strings corresponding to activation functions (see \link{control_rwnn} for possible choices). The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param bias_output TRUE/FALSE: Should a bias be added to the output layer?
#' @param combine_input TRUE/FALSE: Should the input and hidden layer be combined for the output of each hidden layer?
#' @param include_data TRUE/FALSE: Should the original data be included in the returned object? Note: this should almost always be set to 'TRUE', but can be useful and more memory efficient when bagging or boosting an RWNN.
#' @param N_simulations The number of Metropolis-Hastings samples drawn from the posterior.
#' @param N_resample The number of samples drawn during re-sampling after Metropolis-Hastings.
#' @param N_burnin The number of samples used as burn-in during Metropolis-Hastings sampling.
#' @param tau The step-size used when generating new weights from the previous step in Metropolis-Hastings sampler.
#' @param method String indicating the method used after Metropolis-Hastings sampling (see details). 
#' @param trace Show trace every \code{trace} iterations.
#' 
#' @details The current choices of \code{method} are:
#' \describe{
#'     \item{\code{"map"}}{Only the maximum a posterior (MAP) estimate of the weights is returned as an \link{RWNN-object}.}
#'     \item{\code{"stack"}}{The weights are re-sampled (with replacement) weighted using the unnormalised posterior, creating a stacking based \link{ERWNN-object}.}
#'     \item{\code{"posterior"}}{The posterior distributions of the weights are returned.}
#' }
#' 
#' @return A list of control variables.
#' @export
control_sample_rwnn <- function(N_hidden, lnorm = NULL,
                               bias_hidden = TRUE, activation = NULL, 
                               bias_output = TRUE, combine_input = FALSE, include_data = TRUE,
                               N_simulations = 4000, N_burnin = 1000, N_resample = 100, 
                               tau = 0.01, method = NULL, trace = NULL) {
    if (is.null(N_simulations) | !is.numeric(N_simulations)) {
        stop("'N_simulations' has to be numeric.")
    } 
    
    if (N_simulations < 1) {
        stop("'N_simulations' has to be larger than 0.")
    } 
    
    if (is.null(N_burnin) | !is.numeric(N_burnin)) {
        stop("'N_burnin' has to be numeric.")
    } 
    
    if (N_simulations < N_burnin) {
        stop("'N_burnin' has be smaller than 'N_simulations'.")
    }
    
    if (is.null(method) | !is.character(method)) {
        method <- "stack"
    }
    
    if (method %in% c("m", "map", "maxap", "maximumaposterior")) {
        method <- "map"
    }
    else if (method %in% c("stack", "stacking", "averageing")) {
        method <- "stack"
    }
    else if (method %in% c("post", "posterior")) {
        method <- "posterior"
    }
    else {
        stop("The argument supplied to 'method' is not implemented, please set method to 'map', 'stack', or 'posterior'.")
    }
        
    if (method == "stack") {
        if (is.null(N_resample) | !is.numeric(N_resample)) {
            stop("'N_resample' has to be numeric.")
        }
        
        if (N_resample > (N_simulations - N_burnin)) {
            stop("'N_resample' has be smaller than 'N_simulations - N_burnin'.")
        }
    }
    
    if (is.null(trace) | !is.numeric(trace)) {
        trace <- 0
    }
    
    if (is.null(tau) | !is.numeric(tau)) {
        tau <- 0.01
    } 
    
    if (length(N_hidden) < 1) {
        stop("When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see ?lm.")
    }
    
    if (!is.numeric(N_hidden)) {
        stop("Not all elements of the 'N_hidden' vector were numeric.")
    }
    
    if (is.null(lnorm) | !is.character(lnorm)) {
        lnorm <- "l2"
    }
    
    lnorm <- tolower(lnorm)
    if (!(lnorm %in% c("l1", "l2"))) {
        stop("'lnorm' has to be either 'l1' or 'l2'.")
    }
    
    if (length(bias_hidden) == 1) {
        bias_hidden <- rep(bias_hidden, length(N_hidden))
    } else if (length(bias_hidden) == length(N_hidden)) {
        bias_hidden <- bias_hidden
    } else {
        stop("The 'bias_hidden' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")
    }
    
    if (!is.logical(bias_output)) {
        stop("'bias_output' has to be 'TRUE'/'FALSE'.")
    }
    
    
    if (!is.logical(combine_input)) {
        stop("'combine_input' has to be 'TRUE'/'FALSE'.")
    }
    
    
    if (!is.logical(include_data)) {
        stop("'include_data' has to be 'TRUE'/'FALSE'.")
    }
    
    if (is.null(activation) | !is.character(activation)) {
        activation <- "sigmoid"
    }
    
    if (length(activation) == 1) {
        activation <- rep(tolower(activation), length(N_hidden))
    } else if (length(activation) == length(N_hidden)) {
        activation <- tolower(activation)
    } else {
        stop("The 'activation' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")
    }
    
    if (all(!(activation %in% c("sigmoid", "tanh", "relu", "silu", "softplus", "softsign", "sqnl", "gaussian", "sqrbf", "bentidentity", "identity")))) {
        stop("Invalid activation function detected in 'activation' vector. The implemented activation functions are: 'sigmoid', 'tanh', 'relu', 'silu', 'softplus', 'softsign', 'sqnl', 'gaussian', 'sqrbf', 'bentidentity', and 'identity'.")
    }
    
    return(list(lnorm = lnorm, bias_hidden = bias_hidden, activation = activation, 
                bias_output = bias_output, combine_input = combine_input, include_data = include_data,
                N_simulations = N_simulations, N_burnin = N_burnin, N_resample = N_resample, 
                tau = tau, method = method, trace = trace))
}

#' @title Sampling random weight neural networks
#' 
#' @description Uses sampling to pre-train sampling distribution of the hidden layers in random weight neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of the RWNN.
#' @param control A list of additional arguments passed to the \link{control_sample_rwnn} function (includes arguments passed to the \link{control_rwnn} function.).
#' 
#' @return The return object will depend on the choice of \code{method} passed through the \code{control} argument: 
#' \describe{
#'     \item{\code{"map"}}{An \link{RWNN-object}.}
#'     \item{\code{"stack"}}{An \link{ERWNN-object}.}
#'     \item{\code{"posterior"}}{An \link{SRWNN-object}.}
#' }
#' 
#' @export
sample_rwnn <- function(X, y, N_hidden = c(), lambda = NULL, control = list()) {
    UseMethod("sample_rwnn")
}

#' @rdname sample_rwnn
#' @method sample_rwnn default
#' 
#' @example inst/examples/samplerwnn_example.R
#' 
#' @export
sample_rwnn.default <- function(X, y, N_hidden = c(), lambda = NULL, control = list()) {
    ## Creating control object 
    control$N_hidden <- N_hidden
    control <- do.call(control_sample_rwnn, control)
    
    ## Checks
    dc <- data_checks(y, X)
    
    ## Simulation
    sampled_weights_mh <- metropolis_hastings_sampler(
        y = y, X = X, N_hidden = N_hidden, 
        lambda = lambda, control = control
    )
    
    sampled_weights <- sampled_weights_mh$SampledWeights
    loglikelihoods <- sampled_weights_mh$LogLikelihood
    beta <- sampled_weights_mh$Beta
    sigma <- sampled_weights_mh$Sigma
    
    if (control$method == "map") {
        W_map <- sampled_weights[[which.max(loglikelihoods)]]
        object <- list(
            data = if(control$include_data) list(X = X, y = y) else NULL, 
            N_hidden = N_hidden, 
            activation = control$activation, 
            lambda = 0,
            Bias = list(Hidden = control$bias_hidden, Output = control$bias_output),
            Weights = list(Hidden = W_map, Output = beta),
            Sigma = list(Hidden = NA, Output = sigma),
            Combined = control$combine_input
        )
        
        class(object) <- "RWNN"
    } else if (control$method == "stack") {
        ##
        N_simulations <- control$N_simulations
        N_burnin <- control$N_burnin
        N_resample <- control$N_resample
        
        ## Re-sampling
        p <- exp(loglikelihoods - min(loglikelihoods))
        p <- p / sum(p)
        
        resample <- sample(x = N_simulations - N_burnin, size = N_resample, 
                           replace = TRUE, prob = p)
        sampled_weights_re <- sampled_weights[resample]
        
        ## RWNN objects
        objects <- vector("list", N_resample) 
        for (i in seq_len(N_resample)) {
            W_i <- sampled_weights_re[[i]]
            O_i <- last_hidden_layer(X = X, N_hidden = N_hidden, W = W_i, control = control)
            
            theta_i <- estimate_output_weights(O_i, y, control$lnorm, lambda)
            beta_i <- theta_i$beta
            sigma_i <- theta_i$sigma
            
            object_i <- list(
                data = NULL, 
                N_hidden = N_hidden, 
                activation = control$activation, 
                lambda = lambda,
                Bias = list(Hidden = control$bias_hidden, Output = control$bias_output),
                Weights = list(Hidden = W_i, Output = beta_i),
                Sigma = list(Hidden = NA, Output = sigma_i),
                Combined = control$combine_input
            )
            
            class(object_i) <- "RWNN"
            objects[[i]] <- object_i
        }
        
        ## 
        object <- list(
            data = list(X = X, y = y), 
            RWNNmodels = objects, 
            weights = rep(1 / N_resample, N_resample), 
            method = "resample"
        )  
        
        class(object) <- "ERWNN"
    } else if (control$method == "posterior") {
        N_simulations <- control$N_simulations
        N_burnin <- control$N_burnin
        
        p <- exp(loglikelihoods - min(loglikelihoods))
        p <- p / sum(p)
        
        betas <- vector("list", N_simulations - N_burnin)
        sigmas <- rep(NA, N_simulations - N_burnin)
        for (i in seq_len(N_simulations - N_burnin)) {
            W_i <- sampled_weights[[i]]
            O_i <- last_hidden_layer(X = X, N_hidden = N_hidden, W = W_i, control = control)
            
            theta_i <- estimate_output_weights(O_i, y, control$lnorm, lambda)
            betas[[i]] <- theta_i$beta
            sigmas[i] <- theta_i$sigma
        }
        
        object <- list(
            data = list(X = X, y = y), 
            N_hidden = N_hidden, 
            activation = control$activation, 
            lambda = lambda,
            Bias = list(Hidden = control$bias_hidden, Output = control$bias_output),
            Samples = list(W = sampled_weights, Beta = betas, Sigma = sigmas),
            Weights = p, 
            method = "posterior",
            Combined = control$combine_input
        )
        
        class(object) <- "SRWNN"
    } 
    
    return(object)
}

