#' @title abc_rwnn control function
#' 
#' @description A function used to create a control-object for the \link{abc_rwnn} function.
#' 
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lnorm A string indicating the regularisation norm used when estimating the weights in the output layer (either \code{"l1"} or \code{"l2"}).
#' @param bias_hidden A vector of TRUE/FALSE values. The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param activation A vector of strings corresponding to activation functions (see \link{control_rwnn} for possible choices). The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param bias_output TRUE/FALSE: Should a bias be added to the output layer?
#' @param combine_input TRUE/FALSE: Should the input and hidden layer be combined for the output of each hidden layer?
#' @param include_data TRUE/FALSE: Should the original data be included in the returned object? Note: this should almost always be set to 'TRUE', but can be useful and more memory efficient when bagging or boosting an RWNN.
#' @param rng A string indicating the sampling distribution used for generating the weights of the hidden layer (defaults to \code{"runif"}). 
#' @param rng_pars A list of parameters passed to the \code{rng} function (defaults to \code{list(lower = -1, upper = 1)}).   
#' @param N_simulations The number of ABC samples drawn from the posterior.
#' @param N_max The maximum number of simulations performed (even if \code{N_simulations} are not reached).
#' @param metric A metric function used to compare measured and predicted targets (defaults to the L2-norm). The function needs to take two vectors as input and produce a numeric as output. 
#' @param epsilon The maximal metric error allowed for the sample to be kept.
#' @param trace Show trace every \code{trace} iterations.
#' 
#' @return A list of control variables.
#' @export
control_abc_rwnn <- function(N_hidden, lnorm = NULL,
                             bias_hidden = TRUE, activation = NULL, 
                             bias_output = TRUE, combine_input = FALSE, include_data = TRUE,
                             rng = "runif", rng_pars = list(min = -5, max = 5),
                             N_simulations = 1000, N_max = 10000, metric = NULL,
                             epsilon = 0.01, trace = NULL) {
    if (is.null(N_simulations) | !is.numeric(N_simulations)) {
        stop("'N_simulations' has to be numeric.")
    } 
    
    if (N_simulations < 1) {
        stop("'N_simulations' has to be larger than 0.")
    } 
    
    if (is.null(N_max) | !is.numeric(N_max)) {
        stop("'N_max' has to be numeric.")
    } 
    
    if (N_max < N_simulations) {
        stop("'N_max' has to be larger than 'N_simulations'.")
    } 
    
    if (!is.function(metric)) {
        if (!is.null(metric)) {
            warning("The provided 'metric' was not a function. It is set to the average Euclidean distance.")
        }
        
        metric <- function(x, y) {
            return(sqrt(sum((x - y)^2)))
        }
    }
    
    if (is.function(metric)) {
        test_metric <- metric(c(1, 2), c(2, 3))
        if (length(test_metric) > 1) {
            stop("A test was run on the provided 'metric' function resulting in an output of length > 1.")
        }
        else if (is.na(test_metric) || is.nan(test_metric)) {
            stop("A test was run on the provided 'metric' function resulting in an NA, or NaN, value.")
        }
    }
    
    if (is.null(epsilon) | !is.numeric(epsilon)) {
        epsilon <- 0.01
    } 
    
    if (is.null(trace) | !is.numeric(trace)) {
        trace <- 0
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
    
    rng_arg <- formalArgs(rng)[-which(formalArgs(rng) == "n")]
    if (!all(rng_arg %in% names(rng_pars))) {
        stop(paste("The following arguments were not found in 'rng_pars' list:", paste(rng_arg[!(rng_arg %in% names(rng_pars))], collapse = ", ")))
    }
    
    return(list(lnorm = lnorm, bias_hidden = bias_hidden, activation = activation, 
                bias_output = bias_output, combine_input = combine_input, include_data = include_data,
                rng = rng, rng_pars = rng_pars, N_simulations = N_simulations, N_max = N_max, 
                metric = metric, epsilon = epsilon, trace = trace))
}

#' @title Approximate Bayesian computation random weight neural networks
#' 
#' @description Uses approximate Bayesian computation to sample the distribution of the hidden layers in random weight neural network models.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layers of the RWNN.
#' @param control A list of additional arguments passed to the \link{control_abc_rwnn} function (includes arguments passed to the \link{control_rwnn} function.).
#' 
#' @export
abc_rwnn <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, control = list()) {
    UseMethod("abc_rwnn")
}

abc_rwnn.matrix <- function(X, y, N_hidden = c(), lambda = NULL, control = list()) {
    ## Control
    control$N_hidden <- N_hidden
    control <- do.call(control_abc_rwnn, control)
    control_rwnn_ <- control[names(control) %in% names(as.list(args(control_rwnn)))]
    
    ## Checks
    dc <- data_checks(y, X)
    
    ## Simulation
    m <- 0
    n <- 0
    
    metric <- control$metric
    epsilon <- control$epsilon
    
    sampled_weights <- vector("list", length(control$N_simulation))
    betas <- vector("list", length(control$N_simulation))
    sigmas <- rep(NA, length(control$N_simulation))
    p <- rep(NA, length(control$N_simulation))
    while ((n < control$N_simulation) && (m < control$N_max)) {
        rwnn_n <- rwnn(X, y, N_hidden = N_hidden, lambda = lambda, control = control_rwnn_)
        metric_n <- metric(predict(rwnn_n), y)
        
        if (metric_n < epsilon) {
            n <- n + 1
            
            sampled_weights[[n]] <- rwnn_n$Weights$Hidden
            betas[[n]] <- rwnn_n$Weights$Output
            sigmas[n] <- rwnn_n$Sigma$Output
            p[n] <- 1 / metric_n
        }
        
        m <- m + 1
    }
    
    p <- p / sum(p)
    
    ## 
    object <- list(
        formula = NULL,
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
    return(object)
}

#' @rdname abc_rwnn
#' @method abc_rwnn formula
#' 
#' @example inst/examples/abcrwnn_example.R
#' 
#' @export
abc_rwnn.formula <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, control = list()) {
    if (is.null(data)) {
        data <- tryCatch(
            expr = {
                model.matrix(formula)
            },
            error = function(e) {
                message("'data' needs to be supplied when using 'formula'.")
            }
        )
        
        data <- as.data.frame(data)
    }
    
    # Re-capture feature names when '.' is used in formula interface
    formula <- terms(formula, data = data)
    formula <- strip_terms(formula)
    
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
    mm <- abc_rwnn.matrix(X, y, N_hidden = N_hidden, lambda = lambda, control = control)
    mm$formula <- formula
    return(mm)
}
