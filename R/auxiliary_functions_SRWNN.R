#########################################################################
####################### SRWNN neural networks AUX #######################
#########################################################################

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
    H <- rwnn_forward(X, W, control$activation, control$bias_hidden)
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

metropolis_hastings_sampler <- function(y, X, N_hidden, lambda, control) {
    ##
    N_simulations <- control$N_simulations
    N_burnin <- control$N_burnin
    tau  <- control$tau
    trace  <- control$trace
    
    ##
    W_old <- generate_random_weights(X, N_hidden, control$bias_hidden)
    O_old <- last_hidden_layer(X = X, N_hidden = N_hidden, W = W_old, control = control)
    
    theta <- estimate_output_weights(O_old, y, control$lnorm, lambda)
    beta <- theta$beta
    sigma <- log(theta$sigma)
    
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
        O_new <- last_hidden_layer(X, N_hidden, W_new, control)
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
        Sigma = exp(sigma)
    )
    
    return(object)
}

#' @title Coefficients of an SRWNN-object
#' 
#' @param object An \link{SRWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The only additional argument is \code{parameter} taking the values \code{"W"}, \code{"beta"}, or \code{"sigma"}.
#' 
#' @return The weights indicated by \code{parameter}.
#' 
#' @rdname coef.SRWNN
#' @method coef SRWNN
#' @export
coef.SRWNN <- function(object, ...) {
    dots <- list(...)
    if (is.null(dots$parameter)) {
        parameter <- "beta"
    }
    else {
        parameter <- tolower(dots$parameter)
    }
    
    if (parameter == "w") {
        return(object$Samples$W)
    }
    else if (parameter == "beta") {
        return(object$Samples$Beta)
    }
    else if (parameter == "sigma") {
        return(object$Samples$Sigma)
    }
    else {
        stop("The value of 'parameter' was not valid, see '?coef.SRWNN' for valid options of 'parameter'.")
    }
}

#' @title Predicting targets of an SRWNN-object
#' 
#' @param object An \link{SRWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The additional argument '\code{newdata}' and '\code{type}' can be specified, as follows:
#' \describe{
#'   \item{\code{newdata}}{Expects a matrix the same number of features (columns) as in the original data.}
#'   \item{\code{N_samples}}{The number of samples drawn from the posterior.}
#'   \item{\code{type}}{Takes values \code{"all"}, \code{"mean"} (default), and \code{"ci"}, returning a matrix of predictions for all posterior samples, a vector of mean prediction across samples, and a matrix of prediction intervals across the samples.}
#'   \item{\code{p}}{A numeric indicating the credibility level when \code{type = "ci"}.}
#' }
#'
#' @return Either a vector or matrix dependent '\code{type}' (see details). 
#' 
#' @rdname predict.SRWNN
#' @method predict SRWNN
#' @export
predict.SRWNN <- function(object, ...) {
    dots <- list(...)
    
    ##
    type <- dots$type
    if (is.null(type) | !is.character(type)) {
        type <- "mean"
    } else {
        type <- tolower(type)
    }
    
    ##
    N_samples <- dots$N_samples
    if (is.null(N_samples) | !is.numeric(N_samples)) {
        N_samples <- 1000
    }
    
    ##
    p <- dots$p
    if (is.null(p) | !is.numeric(p)) {
        p <- 0.95
    }
    
    alpha <- (1 - p) / 2
    alpha <- c(alpha, 1 - alpha)
    
    ##
    if (is.null(dots$newdata)) {
        newdata <- object$data$X
    } else {
        if (dim(dots$newdata)[2] != (dim(object$Samples$W[[1]][[1]])[1] - as.numeric(object$Bias$Hidden[1]))) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
        
        newdata <- dots$newdata 
    }
    
    ##
    resample <- sample(x = length(object$Weights), size = N_samples, replace = TRUE, prob = object$Weights)
    y_new <- vector("list", N_samples)
    for (b in seq_along(y_new)) {
        newH_b <- rwnn_forward(
            X = newdata, 
            W = object$Samples$W[[resample[b]]], 
            activation = object$activation,
            bias = object$Bias$Hidden
        )
        
        newH_b <- lapply(seq_along(newH_b), function(i) matrix(newH_b[[i]], ncol = object$N_hidden[i]))
        newH_b <- do.call("cbind", newH_b)
        
        ## Estimate parameters in output layer
        if (object$Bias$Output) {
            newH_b <- cbind(1, newH_b)
        }
        
        newO_b <- newH_b
        if (object$Combined) {
            newO_b <- cbind(newH_b, newdata)
        }
        
        y_new[[b]] <- newO_b %*% object$Samples$Beta[[resample[b]]]
    }
    
    y_new <- do.call("cbind", y_new)
    
    ##
    if (type %in% c("a", "all", "f", "full")) {
        return(y_new)
    } else if (type %in% c("m", "mean", "avg", "average")) {
        y_new <- matrix(apply(y_new, 1, mean), ncol = 1)
        return(y_new)
    } else if (type %in% c("ci", "credint", "credibilityinterval")) {
        y_new <- t(apply(y_new, 1, quantile, prob = alpha))
        return(y_new)
    } else {
        stop("The value of 'type' was not valid, see '?predict.SRWNN' for valid options of 'type'.")
    }
}


#' @title Diagnostic-plots of an SRWNN-object
#' 
#' @param x An \link{SRWNN-object}.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments used by the plot.SRWNN function are: 
#' \describe{
#'    \item{\code{parameter}}{A character indicating the parameter, either "W", "beta", or "sigma".}
#'    \item{\code{N_samples}}{The number of samples drawn from the posterior.}
#'    \item{\code{index}}{A vector indicating the layer, neuron, and connection to show when \code{parameter = "W"}.}
#'    \item{\code{breaks}}{The number of breaks used in the histogram when \code{paramter = "W"} or \code{"sigma"} (default: \code{"fd"}).}
#' }
#' 
#' @return NULL
#' 
#' @rdname plot.SRWNN
#' @method plot SRWNN
#'
#' @export
plot.SRWNN <- function(x, ...) {
    dots <- list(...)
    if (is.null(dots$parameter)) {
        parameter <- "beta"
    } else if (tolower(dots$parameter) %in% c("w", "beta", "sigma")) {
        parameter <- tolower(dots$parameter)
    } 
    
    if (is.null(dots$N_samples)) {
        N_samples <- 1000
    } else {
        N_samples <- dots$N_samples[1]
    }
    
    if (is.null(dots$breaks)) {
        breaks <- "fd"
    } else {
        breaks <- dots$breaks
    }
    
    resample <- sample(x = length(x$weights), size = N_samples, replace = TRUE, prob = x$weights)
    if (parameter == "w") {
        if (is.null(dots$index)) {
            warning("The index of chosen 'W' was not found setting it to c(1, 1, 1).")
            index <- c(1, 1, 1)
        } else {
            index <- dots$index
            
            if (length(index) != 3) {
                stop("The index has to be a vector of size 3, indicating the layer, neuron, and connect (in that order).")
            }
        }
        
        resampled_parameter <- x$samples$W[resample]
        resampled_parameter <- sapply(resampled_parameter, function(yy) yy[[index[1]]][index[3], index[2]])
        
        ln <- paste0(index[3], ",", index[2])
        dev.hold()
        hist(resampled_parameter, breaks = breaks, xlab = bquote("W"[.(ln)]), main = bquote(bold("Histogram of W"[.(ln)])))
        dev.flush()
    } else if (parameter == "sigma") {
        resampled_parameter <- x$samples$Sigma[resample]
        
        dev.hold()
        hist(resampled_parameter, breaks = breaks, xlab = bquote(sigma), main = bquote(bold("Histogram of"~sigma)))
        dev.flush()
    } else if (parameter == "beta") {
        resampled_parameter <- t(do.call("cbind", x$samples$Beta[resample]))
        
        dev.hold()
        boxplot(resampled_parameter, xlab = bquote(bold(beta)), main = bquote(bold("Boxplots of"~beta)))
        dev.flush()
    }
    
    return(invisible(NULL))
}
