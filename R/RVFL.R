####################################################################
####################### RVFL neural networks #######################
####################################################################

#### Motor of the RVFL framework ----

#' @title RVFL control function
#' 
#' @description A function used to create a control-object for the \link{RVFL} function.
#' 
#' @param bias_hidden A vector of TRUE/FALSE values. The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param activation A vector of strings corresponding to activation functions (see details for possible choices). The vector should have length 1, or the length should be equal to the number of hidden layers.
#' @param bias_output TRUE/FALSE: Should a bias be added to the output layer?
#' @param combine_input TRUE/FALSE: Should the input and hidden layer be combined for the output of each hidden layer?
#' @param include_data TRUE/FALSE: Should the original data be included in the returned object? Note: this should almost always be set to 'TRUE', but can be useful and more memory efficient when bagging or boosting an RVFL.
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
control_RVFL <- function(bias_hidden = TRUE, activation = NULL, 
                         bias_output = TRUE, combine_input = FALSE, 
                         include_data = TRUE) {
    return(list(bias_hidden = bias_hidden, activation = activation, 
                bias_output = bias_output, combine_input = combine_input, 
                include_data = include_data))
}

#' @title Random vector functional link
#' 
#' @description Set-up and estimate weights of a random vector functional link neural network.
#' 
#' @param X A matrix of observed features used to train the parameters of the output layer.
#' @param y A vector of observed targets used to train the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layer.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments are all passed to the \link{control_RVFL}-function.
#' 
#' @return An RVFL-object containing the random and fitted weights of the RVFL-model. An RVFL-object contains the following:
#' \describe{
#'     \item{\code{data}}{The original data used to estimate the weights.}
#'     \item{\code{N_hidden}}{The vector of neurons in each layer.}
#'     \item{\code{activation}}{The vector of the activation functions used in each layer.}
#'     \item{\code{Bias}}{The \code{TRUE/FALSE} bias vectors set by the control function for both hidden layers, and the output layer.}
#'     \item{\code{Weights}}{The weigths of the neural network, split into random (stored in hidden) and estimated (stored in output) weights.}
#'     \item{\code{Sigma}}{The standard deviation of the corresponding linear model.}
#'     \item{\code{Combined}}{A \code{TRUE/FALSE} stating whether the direct links were made to the input.}
#' }
#' 
#' @export
RVFL <- function(X, y, N_hidden, lambda = 0, ...) {
    UseMethod("RVFL")
}

#' @rdname RVFL
#' @method RVFL default
#' 
#' @example inst/examples/rvfl_example.R
#' 
#' @export
RVFL.default <- function(X, y, N_hidden, lambda = 0, ...) {
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
    }
    else if (length(control$bias_hidden) == length(N_hidden)) {
        bias_hidden <- control$bias_hidden
    }
    else {
        stop("The 'bias_hidden' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")
    }
    
    activation <- control$activation
    if (is.null(activation)) {
        activation <- "sigmoid"
    }
    
    if (length(activation) == 1) {
        activation <- rep(tolower(activation), length(N_hidden))
    }
    else if (length(activation) == length(N_hidden)) {
        activation <- tolower(activation)
    }
    else {
        stop("The 'activation' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")
    }
    
    if (all(!(activation %in% c("sigmoid", "tanh", "relu", "silu", "softplus", "softsign", "sqnl", "gaussian", "sqrbf", "bentidentity", "identity")))) {
        stop("Invalid activation function detected in 'activation' vector. The implemented activation functions are: 'sigmoid', 'tanh', 'relu', 'silu', 'softplus', 'softsign', 'sqnl', 'gaussian', 'sqrbf', 'bentidentity', and 'identity'.")
    }
    
    if (is.null(lambda)) {
        lambda <- 0
        warning("Note: 'lambda' was not supplied and set to 0.")
    }
    else if (lambda < 0) {
        lambda <- 0
        warning("'lambda' has to be a real number larger than or equal to 0.")
    }
    
    if (length(lambda) > 1) {
        lambda <- lambda[1]
        warning("The length of 'lambda' was larger than 1; continuing analysis using only the first element.")
    }
    
    ## Initialisation
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
    
    W_output <- estimate_output_weights(O, y, lambda)
    
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

#### Auxiliary ----

#' @title Coefficients of the RVFL object.
#' 
#' @param object An RVFL-object.
#' @param ... Additional arguments.
#' 
#' @details No additional arguments are used in this instance.
#' 
#' @return The estimated weights of the output-layer.
#' 
#' @rdname coef.RVFL
#' @method coef RVFL
#' @export
coef.RVFL <- function(object, ...) {
    return(object$Weights$Output)
}

#' @title Predicting targets of an RVFL object.
#' 
#' @param object An RVFL-object.
#' @param ... Additional arguments.
#' 
#' @details The only additional argument used by the function is '\code{newdata}', which expects a matrix with the same number of features (columns) as in the original data.
#' 
#' @return A vector of predicted targets.
#' 
#' @rdname predict.RVFL
#' @method predict RVFL
#' @export
predict.RVFL <- function(object, ...) {
    dots <- list(...)
    
    if (is.null(dots$newdata)) {
        if (is.null(object$data)) {
            stop("The RVFL-object does not contain any data: \nEither supply 'newdata', or re-create object with 'include_data = TRUE' (default).")
        }
        
        newdata <- object$data$X
    }
    else {
        if (dim(dots$newdata)[2] != (dim(object$Weights$Hidden[[1]])[1] - as.numeric(object$Bias$Hidden[1]))) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
        
        newdata <- dots$newdata 
    }
    
    newH <- rvfl_forward(
        X = newdata, 
        W = object$Weights$Hidden, 
        activation = object$activation,
        bias = object$Bias$Hidden
    )
    
    newH <- lapply(seq_along(newH), function(i) matrix(newH[[i]], ncol = object$N_hidden[i]))
    newH <- do.call("cbind", newH)
    
    ## Estimate parameters in output layer
    if (object$Bias$Output) {
        newH <- cbind(1, newH)
    }
    
    newO <- newH
    if (object$Combined) {
        newO <- cbind(newH, newdata)
    }
    
    newy <- newO %*% object$Weights$Output
    return(newy)
}

#' @title Residuals of the RVFL object.
#' 
#' @param object An RVFL-object.
#' @param ... Additional arguments.
#' 
#' @details Besides the arguments passed to the '\code{predict}' function, the argument '\code{type}' can be supplied defining the type of residual returned by the function. Currently only \code{"rs"} (standardised residuals), and \code{"raw"} (default) are implemented.
#'
#' @return A vector of residuals of the desired '\code{type}' (see details). 
#'
#' @rdname residuals.RVFL
#' @method residuals RVFL
#' @export
residuals.RVFL <- function(object, ...) {
    dots <- list(...)
    type <- dots$type
    if (is.null(type)) {
        type <- "raw"
    }
    
    if (is.null(dots$newdata)) {
        if (is.null(object$data)) {
            stop("The RVFL-object does not contain any data: \nEither supply 'newdata', or re-create RVFL-object with 'include_data = TRUE' (default).")
        }
        
        newdata <- object$data$X
    }
    else {
        if (dim(dots$newdata)[2] != (dim(object$Weights$Hidden[[1]])[1] - as.numeric(object$Bias$Hidden[1]))) {
            stop("The number of features (columns) provided in 'newdata' does not match the number of features of the model.")
        }
        
        newdata <- dots$newdata
    }
    
    newy <- predict.RVFL(object, newdata = newdata, ...)
    
    r <- newy - object$data$y
    if (tolower(type) %in% c("standard", "standardised", "rs")) {        
        r <- r / object$Sigma$Output
    }
    
    return(r)
}

#' @title Diagnostic-plots of an RVFL-object.
#' 
#' @param x An RVFL-object.
#' @param ... Additional arguments.
#' 
#' @details The additional arguments used by the function are '\code{X_val}' and '\code{y_val}', i.e. the features and targets of the validation-set. These are helpful when analysing whether overfitting of model has occured.  
#' 
#' @rdname plot.RVFL
#' @method plot RVFL
#'
#' @return NULL
#' 
#' @export
plot.RVFL <- function(x, ...) {
    dots <- list(...)
    if (is.null(dots$X_val) || is.null(dots$y_val)) {
        if (is.null(x$data)) {
            stop("The RVFL-object does not contain any data: \nEither supply 'X_val' and 'y_val', or re-create RVFL-object with 'include_data = TRUE' (default).")
        }
        
        X_val <- x$data$X
        y_val <- x$data$y
    }
    else {
        X_val <- dots$X_val
        y_val <- dots$y_val
    }
    
    y_hat <- predict(x, newdata = X_val)
    
    dev.hold()
    plot(y_hat ~ y_val, pch = 16, 
         xlab = "Observed targets", ylab = "Predicted targets")
    abline(0, 1, col = "dodgerblue", lty = "dashed", lwd = 2)
    dev.flush()
    
    readline(prompt = "Press [ENTER] for next plot...")
    dev.hold()
    plot(I(y_hat - y_val) ~ seq(length(y_val)), pch = 16,
         xlab = "Index", ylab = "Residual") 
    abline(0, 0, col = "dodgerblue", lty = "dashed", lwd = 2)
    dev.flush()
    
    return(invisible(NULL))
}
