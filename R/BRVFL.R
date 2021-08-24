############################################################################
####################### Bayesian RVFL neural network #######################
############################################################################

#### ----
#' @title Sampling random vector functional links
#' 
#' @description Uses sampling to pre-train sampling distribution of the hidden layers in random vector functional link neural network models.
#' 
#' @param X A matrix of observed features used to estimate the parameters of the output layer.
#' @param y A vector of observed targets used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param N_simulations The number of Metropolis-Hastings samples drawn from the posterior.
#' @param N_resample The number of samples drawn during re-sampling after Metropolis-Hastings.
#' @param N_burnin The number of samples used as burn-in during Metropolis-Hastings sampling.
#' @param trace Show trace every \code{trace} iterations.
#' @param ... Additional arguments passed to the \link{control_RVFL} function.
#' 
#' @return An ERVFL-object containing the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{RVFLmodels}}{A list of \link{RVFL}-objects.}
#'     \item{\code{weights}}{A vector of ensemble weights.}
#'     \item{\code{method}}{A string indicating the method ('boosting' in this case)}
#' }
#' 
#' @export
sampleRVFL <- function(X, y, N_hidden, N_simulations = 4000, N_burnin = 1000, N_resample = 100, trace = NULL, ...) {
    UseMethod("sampleRVFL")
}

#' @rdname sampleRVFL
#' @method sampleRVFL default
#' 
#' @example inst/examples/samplervfl_example.R
#' 
#' @export
sampleRVFL.default <- function(X, y, N_hidden, N_simulations = 4000, N_burnin = 1000, N_resample = 100, trace = NULL, ...) {
    ## Creating control object 
    dots <- list(...)
    control <- do.call(control_RVFL, dots)
    
    ## Checks
    # Data
    if (!is.matrix(X)) {
        warning("'X' has to be a matrix... trying to cast 'X' as a matrix.")
        X <- as.matrix(X)
    }
    
    if (!is.matrix(y)) {
        warning("'y' has to be a matrix... trying to cast 'y' as a matrix.")
        y <- as.matrix(y)
    }
    
    if (dim(y)[2] != 1) {
        warning("More than a single column was detected in 'y', only the first column is used in the model.")
        y <- matrix(y[, 1], ncol = 1)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    # Parameters
    if (length(N_hidden) < 1) {
        stop("When the number of hidden layers is equal to 0, this model reduces to a linear model, ?lm.")
    }
    
    if (length(control$bias_hidden) == 1) {
        bias_hidden <- rep(control$bias_hidden, length(N_hidden))
    } else if (length(control$bias_hidden) == length(N_hidden)) {
        bias_hidden <- control$bias_hidden
    } else {
        stop("The 'bias_hidden' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")
    }
    
    # Activation
    activation <- control$activation
    if (is.null(activation)) {
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
    
    # Simulation
    if (N_simulations < 1) {
        stop("'N_simulations' has to be larger than 0.")
    }
    
    if (N_simulations < N_burnin) {
        stop("'N_burnin' has be smaller than 'N_simulations'.")
    }
    
    if (N_resample > (N_simulations - N_burnin)) {
        stop("'N_resample' has be smaller than 'N_simulations - N_burnin'.")
    }
    
    if (is.null(trace)) {
        trace <- 0
    }
    
    ## Independent MH sampling
    # Initialise
    W_old <- generate_random_weights(X, N_hidden, bias_hidden)
    O_old <- last_hidden_layer(X, N_hidden, W_old, activation, bias_hidden, control)
    theta <- RVFL:::estimate_output_weights(O_old, y, 0)
    #beta_old <- update_output_weights(theta_old$beta, 0.1)
    #sigma_old <- update_output_weights(log(theta_old$sigma), 0.1)
    
    ll_old <- model_likelihood(O_old, y, theta$beta, theta$sigma) # beta_old, sigma_old)
    
    # 
    sampled_weights <- vector("list", N_simulations - N_burnin)
    estimated_beta <- vector("list", N_simulations - N_burnin)
    estimated_sigma <- vector("list", N_simulations - N_burnin)
    loglikelihoods <- rep(NA, N_simulations - N_burnin)
    for (i in seq_len(N_simulations)) {
        if ((trace > 0) && ((i == 1) || (i == N_simulations) || ((i %% trace) == 0))) {
            cat(i, "/", N_simulations, "\n")
        }
        
        #
        W_new <- lapply(W_old, function(xx) xx +  matrix(rnorm(length(xx), 0, 0.01), nrow = nrow(xx), ncol = ncol(xx))) # generate_random_weights(X, N_hidden, bias_hidden) 
        O_new <- last_hidden_layer(X, N_hidden, W_new, activation, bias_hidden, control)
        #beta_new <- update_output_weights(beta_old, 0.1)
        #sigma_new <- update_output_weights(sigma_old, 0.1)
        ll_new <- model_likelihood(O_new, y, theta$beta, theta$sigma) # beta_new, sigma_new)
        
        #
        mh <- min(ll_new - ll_old, 1)
        u <- runif(1, 0, 1)
        if (mh > log(u)) {
            W_old <- W_new
            beta_old <- beta_new
            sigma_old <- sigma_new
            ll_old <- ll_new
        }
        
        if (i > N_burnin) {
            sampled_weights[[i - N_burnin]] <- W_old
            estimated_beta[[i - N_burnin]] <- theta$beta # beta_old
            estimated_sigma[[i - N_burnin]] <- theta$sigma # sigma_old
            loglikelihoods[i - N_burnin] <- ll_old
        }
    }
    
    ## Re-sampling
    p <- loglikelihoods - (min(loglikelihoods) - 1e-8)
    p <- p / sum(p)
    
    resample <- sample(x = N_simulations - N_burnin, size = N_resample, replace = FALSE, prob = p)
    sampled_weights_re <- sampled_weights[resample]
    estimated_beta_re <- estimated_beta[resample]
    estimated_sigma_re <- estimated_sigma[resample]
    
    ## RVFL object
    objects <- vector("list", N_resample) 
    for (i in seq_len(N_resample)) {
        object_i <- list(
            data = NULL, 
            N_hidden = N_hidden, 
            activation = activation, 
            lambda = lambda,
            Bias = list(Hidden = bias_hidden, Output = control$bias_output),
            Weights = list(Hidden = sampled_weights_re[[i]], Output = estimated_beta_re[[i]]),
            Sigma = list(Hidden = NA, Output = estimated_sigma_re[[i]]),
            Combined = control$combine_input
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
    return(object)
}

#### Auxiliary ----
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

update_output_weights <- function(par, tau) {
    return(par + rnorm(length(par), 0, tau))
}

last_hidden_layer <- function(X, N_hidden, W, activation, bias_hidden, control) {
    H <- rvfl_forward(X, W, activation, bias_hidden)
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
