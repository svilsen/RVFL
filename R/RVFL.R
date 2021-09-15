####################################################################
####################### RVFL neural networks #######################
####################################################################

#' @title RVFL control function
#' 
#' @description A function used to create a control-object for the \link{RVFL} function.
#' 
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lnorm A string indicating the regularisation norm used when estimating the weights in the output layer (either \code{"l1"} or \code{"l2"}).
#' @param bias_hidden A vector of TRUE/FALSE values. The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param activation A vector of strings corresponding to activation functions (see details for possible choices). The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param bias_output TRUE/FALSE: Should a bias be added to the output layer?
#' @param combine_input TRUE/FALSE: Should the input and hidden layer be combined for the output of each hidden layer?
#' @param include_data TRUE/FALSE: Should the original data be included in the returned object? Note: this should almost always be set to 'TRUE', but can be useful and more memory efficient when bagging or boosting an RVFL.
#' @param N_features The number of randomly chosen features in the RVFL model. Note: This is meant for use in \link{bagRVFL}, and it is recommended that is not be used outside of that function. 
#' @param rng A string indicating the sampling distribution used for generating the weights of the hidden layer (defaults to \code{"runif"}). 
#' @param rng_pars A list of parameters passed to the \code{rng} function (defaults to \code{list(lower = -1, upper = 1)}).   
#' 
#' @details The possible activation functions supplied to '\code{activation}' are:
#' \describe{
#'     \item{\code{"sigmoid"}}{\deqn{a(x) = \frac{1}{1 + \exp(-x)}}}
#'     \item{\code{"tanh"}}{\deqn{a(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}}}
#'     \item{\code{"relu"}}{\deqn{a(x) = \max\{0, x\}}}
#'     \item{\code{"silu"}}{\deqn{a(x) = \frac{x}{1 + \exp(-x)}}}
#'     \item{\code{"softplus"}}{\deqn{a(x) = \log(1 + \exp(x))}}
#'     \item{\code{"softsign"}}{\deqn{a(x) = \frac{x}{1 + |x|)}}}
#'     \item{\code{"sqnl"}}{\deqn{a(x) = -1, if x < -2, a(x) = x + \frac{x^2}{4}, if -2 \le x < 0, a(x) = x - \frac{x^2}{4}, if 0 \le x \le 2, and a(x) = 2, if x > 2}}
#'     \item{\code{"gaussian"}}{\deqn{a(x) = \exp(x^2)}}
#'     \item{\code{"sqrbf"}}{\deqn{a(x) = 0, if |x| \ge 2, a(x) = \frac{(2 - |x|)^2}{2}, if 1 < |x| < 2, and a(x) = 1 - \frac{x^2}{2}, if |x| \le 1}}
#'     \item{\code{"bentidentity"}}{\deqn{a(x) = \frac{\sqrt{x^2 + 1} - 1}{2} + x}}
#'     \item{\code{"identity"}}{\deqn{a(x) = x}}
#' }
#' 
#' @return A list of control variables.
#' @export
control_RVFL <- function(N_hidden, lnorm = NULL,
                         bias_hidden = TRUE, activation = NULL, 
                         bias_output = TRUE, combine_input = FALSE, 
                         include_data = TRUE, N_features = NULL, 
                         rng = "runif", rng_pars = list(min = -1, max = 1)) {
    if (length(N_hidden) < 1) {
        stop("When the number of hidden layers is equal to 0, this model reduces to a linear model, ?lm.")
    }
    
    if (is.null(lnorm)) {
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
    
    rng_arg <- formalArgs(rng)[-which(formalArgs(rng) == "n")]
    if (!all(rng_arg %in% names(rng_pars))) {
        stop(paste("The following arguments were not found in 'rng_pars' list:", paste(rng_arg[!(rng_arg %in% names(rng_pars))], collapse = ", ")))
    }
    
    return(list(lnorm = lnorm, bias_hidden = bias_hidden, activation = activation, 
                bias_output = bias_output, combine_input = combine_input, 
                include_data = include_data, N_features = N_features, 
                rng = rng, rng_pars = rng_pars))
}

#' @title Random vector functional link
#' 
#' @description Set-up and estimate weights of a random vector functional link neural network.
#' 
#' @param X A matrix of observed features used to train the parameters of the output layer.
#' @param y A vector of observed targets used to train the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layer.
#' @param control A list of additional arguments passed to the \link{control_RVFL} function.
#' 
#' @details The function \code{ELM} is a wrapper for the general \code{RVFL} function without the link between features and targets. Furthermore, notice that \code{dRVFL} is handled by increasing the number of elements passed in \code{N_hidden}.
#' 
#' @return An \link{RVFL-object}.
#' 
#' @export
RVFL <- function(X, y, N_hidden, lambda = 0, control = list()) {
    UseMethod("RVFL")
}

#' @rdname RVFL
#' @method RVFL default
#' 
#' @example inst/examples/rvfl_example.R
#' 
#' @export
RVFL.default <- function(X, y, N_hidden, lambda = 0, control = list()) {
    ## Creating control object 
    control$N_hidden <- N_hidden
    control <- do.call(control_RVFL, control)
    
    #
    lnorm <- control$lnorm
    bias_hidden <- control$bias_hidden
    activation <- control$activation
    N_features <- control$N_features
    rng_function <- control$rng
    rng_pars <- control$rng_pars
    
    ## Checks
    dc <- data_checks(y, X)
    
    # Regularisation
    if (is.null(lambda)) {
        lambda <- 0
        warning("Note: 'lambda' was not supplied and set to 0.")
    } else if (lambda < 0) {
        lambda <- 0
        warning("'lambda' has to be a real number larger than or equal to 0.")
    }
    
    if (length(lambda) > 1) {
        lambda <- lambda[1]
        warning("The length of 'lambda' was larger than 1; continuing analysis using only the first element.")
    }
    
    if (is.null(N_features)) {
        N_features <- dim(X)[2]
    }
    
    if ((N_features < 1) || (N_features > dim(X)[2])) {
        stop("'N_features' have to be in the interval [1; dim(X)[2]].")
    }
    
    ## Creating random weights
    X_dim <- dim(X)
    W_hidden <- vector("list", length = length(N_hidden))
    for (w in seq_along(W_hidden)) {
        if (w == 1) {
            nr_connections <- N_hidden[w] * (X_dim[2] + as.numeric(bias_hidden[w]))
        }
        else {
            nr_connections <- N_hidden[w] * (N_hidden[w - 1] + as.numeric(bias_hidden[w]))
        }
        
        rng_pars$n <- nr_connections
        random_weights <- do.call(rng_function, rng_pars)
        W_hidden[[w]] <- matrix(random_weights, ncol = N_hidden[w]) 
        
        if ((w == 1) && (N_features < dim(X)[2])) {
            indices_f <- sample(ncol(X), N_features, replace = FALSE) + as.numeric(bias_hidden[w])
            W_hidden[[w]][-indices_f, ] <- 0
        }
    }
    
    ## Values of last hidden layer
    H <- rvfl_forward(X, W_hidden, activation, bias_hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = N_hidden[i]))
    H <- do.call("cbind", H)
    
    ## Estimate parameters in output layer
    if (control$bias_output) {
        H <- cbind(1, H)
    }
    
    O <- H
    if (control$combine_input) {
        O <- cbind(X, H)
    }
    
    W_output <- estimate_output_weights(O, y, lnorm, lambda)
    
    ## Return object
    object <- list(
        data = if(control$include_data) list(X = X, y = y) else NULL, 
        N_hidden = N_hidden, 
        activation = activation, 
        lambda = lambda,
        Bias = list(Hidden = bias_hidden, Output = control$bias_output),
        Weights = list(Hidden = W_hidden, Output = W_output$beta),
        Sigma = list(Hidden = NA, Output = W_output$sigma),
        Combined = control$combine_input
    )
    
    class(object) <- "RVFL"
    return(object)
}

#' @rdname RVFL
#' 
#' @export
ELM <- function(X, y, N_hidden, lambda = 0, control = list()) {
    control$N_hidden <- N_hidden
    control$combine_input <- FALSE
    
    elm_object <- list(X = X, y = y, N_hidden = N_hidden, lambda = lambda, control = control)
    object <- do.call(RVFL, elm_object)
    return(object)
}
    