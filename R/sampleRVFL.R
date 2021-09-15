################################################################################
####################### Re-sampling ERVFL neural network #######################
################################################################################

generate_random_weights <- function(X, N_hidden, bias_hidden) { 
    X_dim <- dim(X)
    W_hidden <- vector("list", length = length(N_hidden))
    for (w in seq_along(W_hidden)) {
        if (w == 1) {
            nr_connections <- N_hidden[w] * (X_dim[2] + as.numeric(bias_hidden[w]))
        }
        else {
            nr_connections <- N_hidden[w] * (N_hidden[w - 1] + as.numeric(bias_hidden[w]))
        }
        
        random_weights <- runif(nr_connections, -1, 1)
        W_hidden[[w]] <- matrix(random_weights, ncol = N_hidden[w]) 
    }
    return(W_hidden)
}

update_random_weights <- function(W_old, tau) {
    N <- length(W_old)
    
    W_new <- vector("list", N)
    for (n in seq_len(N)) {
        W_old_n <- W_old[[n]]
        W_new[[n]] <- W_old_n + matrix(rnorm(length(W_old_n), 0, tau), nrow = nrow(W_old_n), ncol = ncol(W_old_n))
    }
    
    return(W_new)
}

last_hidden_layer <- function(X, N_hidden, W, control) {
    H <- rvfl_forward(X, W, control$activation, control$bias_hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = N_hidden[i]))
    H <- do.call("cbind", H)
    
    if (control$bias_output) {
        H <- cbind(1, H)
    }
    
    O <- H
    if (control$combine_input) {
        O <- cbind(X, H)
    }
    
    return(O)
}

model_likelihood <- function(O, y, beta, sigma) {
    yhat <- O %*% beta
    ll <- sum(dnorm(y, mean = yhat, sd = exp(sigma), log = TRUE))
    return(ll)
}

metropolis_hastings_sampler <- function(y, X, N_hidden, lambda, control_rvfl, control_sample) {
    ##
    N_simulations <- control_sample$N_simulations
    N_burnin <- control_sample$N_burnin
    tau  <- control_sample$tau
    trace  <- control_sample$trace
    
    ##
    W_old <- generate_random_weights(X, N_hidden, control_rvfl$bias_hidden)
    O_old <- last_hidden_layer(X = X, N_hidden = N_hidden, W = W_old, control = control_rvfl)
    
    theta <- estimate_output_weights(O_old, y, control_rvfl$lnorm, lambda)
    beta <- theta$beta
    sigma <- theta$sigma
    
    ll_old <- model_likelihood(O_old, y, beta, sigma) 
    
    ##
    sampled_weights <- vector("list", N_simulations - N_burnin)
    loglikelihoods <- rep(NA, N_simulations - N_burnin)
    for (i in seq_len(N_simulations)) {
        if ((trace > 0) && ((i == 1) || (i == N_simulations) || ((i %% trace) == 0))) {
            cat(i, "/", N_simulations, "\n")
        }
        
        #
        W_new <- update_random_weights(W_old, tau) 
        O_new <- last_hidden_layer(X, N_hidden, W_new, control_rvfl)
        ll_new <- model_likelihood(O_new, y, beta, sigma)
        
        #
        mh <- min(ll_new - ll_old, 1)
        u <- runif(1, 0, 1)
        if (mh > log(u)) {
            W_old <- W_new
            ll_old <- ll_new
        }
        
        if (i > N_burnin) {
            sampled_weights[[i - N_burnin]] <- W_old
            loglikelihoods[i - N_burnin] <- ll_old
        }
    }
    
    object <- list(
        SampledWeights = sampled_weights, 
        LogLikelihood = loglikelihoods, 
        Beta = beta,
        Sigma = sigma
    )
    
    return(object)
}

#' @title sampleRVFL control function.
#' 
#' @description A function used to create a control-object for the \link{sampleRVFL} function.
#' 
#' @param N_simulations The number of Metropolis-Hastings samples drawn from the posterior.
#' @param N_resample The number of samples drawn during re-sampling after Metropolis-Hastings.
#' @param N_burnin The number of samples used as burn-in during Metropolis-Hastings sampling.
#' @param tau The step-size used when generating new weights from the previous step in Metropolis-Hastings sampler.
#' @param method String indicating the method used after Metropolis-Hastings sampling (see details). 
#' @param trace Show trace every \code{trace} iterations.
#' 
#' @details The current choices of \code{method} are:
#' \describe{
#'     \item{\code{"map"}}{Only the maximum a posterior (MAP) estimate of the weights is returned.}
#'     \item{\code{"resampling"}}{The weights are resampled (with replacement) weighted using the unnormalised posterior.}
#' }
#' 
#' @return A list of control variables.
#' @export
control_sampleRVFL <- function(N_simulations = 4000, N_burnin = 1000, N_resample = 100, 
                               tau = 0.01, method = NULL, trace = NULL) {
    if (N_simulations < 1) {
        stop("'N_simulations' has to be larger than 0.")
    } 
    
    if (N_simulations < N_burnin) {
        stop("'N_burnin' has be smaller than 'N_simulations'.")
    }
    
    if (is.null(method)) {
        method <- "resample"
    }
    
    if (method == "resample") {
        if (N_resample > (N_simulations - N_burnin)) {
            stop("'N_resample' has be smaller than 'N_simulations - N_burnin'.")
        }
    }
    
    if (is.null(trace)) {
        trace <- 0
    }
    
    if (is.null(tau)) {
        tau <- 0.01
    } 
    
    return(list(N_simulations = N_simulations, N_burnin = N_burnin, N_resample = N_resample, 
                tau = tau, method = method, trace = trace))
}

#' @title Sampling random vector functional links
#' 
#' @description Uses sampling to pre-train sampling distribution of the hidden layers in random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of the RVFL.
#' @param control_rvfl A list of additional arguments passed to the \link{control_RVFL} function.
#' @param control_sample A list of additional arguments passed to the \link{control_sampleRVFL} function.
#' 
#' @return The return object will depend on the choice of \code{method} passed through the \code{control_sample} argument. 
#' If \code{method} is set to \code{"map"} an \link{RVFL-object} is returned, while an \link{ERVFL-object} is returned when \code{method} is set to \code{"resample"}.
#' 
#' @export
sampleRVFL <- function(X, y, N_hidden, control_rvfl = list(), control_sample = list()) {
    UseMethod("sampleRVFL")
}

#' @rdname sampleRVFL
#' @method sampleRVFL default
#' 
#' @example inst/examples/samplervfl_example.R
#' 
#' @export
sampleRVFL.default <- function(X, y, N_hidden, lambda, control_rvfl = list(), control_sample = list()) {
    ## Creating control object 
    control_rvfl$N_hidden <- N_hidden
    control_rvfl <- do.call(control_RVFL, control_rvfl)
    control_sample <- do.call(control_sampleRVFL, control_sample)
    
    ## Checks
    dc <- data_checks(y, X)
    
    ## Simulation
    sampled_weights_mh <- metropolis_hastings_sampler(
        y = y, X = X, N_hidden = N_hidden, lambda = lambda,
        control_rvfl = control_rvfl, 
        control_sample = control_sample
    )
    
    sampled_weights <- sampled_weights_mh$SampledWeights
    loglikelihoods <- sampled_weights_mh$LogLikelihood
    beta <- sampled_weights_mh$Beta
    sigma <- sampled_weights_mh$Sigma
    
    if (control_sample$method == "map") {
        W_map <- sampled_weights[[which.max(loglikelihoods)]]
        object <- list(
            data = if(control_rvfl$include_data) list(X = X, y = y) else NULL, 
            N_hidden = N_hidden, 
            activation = control_rvfl$activation, 
            lambda = 0,
            Bias = list(Hidden = control_rvfl$bias_hidden, Output = control_rvfl$bias_output),
            Weights = list(Hidden = W_map, Output = beta),
            Sigma = list(Hidden = NA, Output = sigma),
            Combined = control_rvfl$combine_input
        )
        
        class(object) <- "RVFL"
    } else if (control_sample$method == "resample") {
        ##
        N_simulations <- control_sample$N_simulations
        N_burnin <- control_sample$N_burnin
        N_resample <- control_sample$N_resample
        
        ## Re-sampling
        p <- exp(loglikelihoods - min(loglikelihoods))
        p <- p / sum(p)
        
        resample <- sample(x = N_simulations - N_burnin, 
                           size = N_resample, replace = TRUE, prob = p)
        sampled_weights_re <- sampled_weights[resample]
        
        ## RVFL objects
        objects <- vector("list", N_resample) 
        for (i in seq_len(N_resample)) {
            object_i <- list(
                data = NULL, 
                N_hidden = N_hidden, 
                activation = control_rvfl$activation, 
                lambda = 0,
                Bias = list(Hidden = control_rvfl$bias_hidden, Output = control_rvfl$bias_output),
                Weights = list(Hidden = sampled_weights_re[[i]], Output = beta),
                Sigma = list(Hidden = NA, Output = sigma),
                Combined = control_rvfl$combine_input
            )
            
            class(object_i) <- "RVFL"
            objects[[i]] <- object_i
        }
        
        ## 
        object <- list(
            data = list(X = X, y = y), 
            RVFLmodels = objects, 
            weights = rep(1 / N_resample, N_resample), 
            method = "resample"
        )  
        
        class(object) <- "ERVFL"
    } else {
        stop("The supplied method is not implemented, please set method to either 'map' or 'resample'.")
    }
    
    return(object)
}

