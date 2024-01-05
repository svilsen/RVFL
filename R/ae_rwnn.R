##################################################################################
####################### AE pre-trained RWNN neural network #######################
##################################################################################

#' @title Auto-encoder pre-trained random weight neural networks
#' 
#' @description Set-up and estimate weights of a random weight neural network using an auto-encoder for unsupervised pre-training of the hidden weights.
#' 
#' @param formula A \link{formula} specifying features and targets used to estimate the parameters of the output layer. 
#' @param data A data-set (either a \link{data.frame} or a \link[tibble]{tibble}) used to estimate the parameters of the output layer.
#' @param N_hidden A vector of integers designating the number of neurons in each of the hidden layers (the length of the list is taken as the number of hidden layers).
#' @param lambda The penalisation constant used when training the output layer.
#' @param method The penalisation type used in the auto-encoder (either \code{"l1"} or \code{"l2"}).
#' @param type A string indicating whether this is a regression or classification problem. 
#' @param control A list of additional arguments passed to the \link{control_rwnn} function.
#' 
#' @return An \link{RWNN-object}.
#' 
#' @export
ae_rwnn <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, method = "l1", type = NULL, control = list()) {
    UseMethod("ae_rwnn")
}

ae_rwnn.matrix <- function(X, y, N_hidden = c(), lambda = NULL, method = "l1", type = NULL, control = list()) {
    ## Creating control object 
    if (length(N_hidden) > 1) {
        N_hidden <- N_hidden[1]
        warning("More than one hidden-layer was found, however, this method is only designed for a single hidden-layer, therefore, only the first element 'N_hidden' is used.")
    }
    
    control$N_hidden <- N_hidden
    control <- do.call(control_rwnn, control)
    
    #
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
        warning("Note: 'lambda' is set to '0', as it was not supplied.")
    } else if (lambda < 0) {
        lambda <- 0
        warning("'lambda' has to be a real number larger than or equal to '0'.")
    }
    
    if (length(lambda) > 1) {
        lambda <- lambda[1]
        warning("The length of 'lambda' was larger than 1; only the first element will be used.")
    }
    
    if (is.null(N_features)) {
        N_features <- ncol(X)
    }
    
    if (length(N_features) > 1) {
        N_features <- N_features[1]
        warning("The length of 'N_features' was larger than 1; only the first element will be used.")
    }
    
    if ((N_features < 1) || (N_features > dim(X)[2])) {
        stop("'N_features' have to be between '1' and the total number of features.")
    }
    
    ## Creating random weights
    nr_rows <- (X_dim[2] + as.numeric(bias_hidden[1]))
    
    if (is.character(rng_function)) {
        if (rng_function %in% c("o", "orto", "orthogonal")) {
            random_weights <- (rng_pars$max - rng_pars$min) * random_orthonormal(1, nr_rows, X, W_hidden, N_hidden, activation, bias_hidden) + rng_pars$min
        }
        else if (rng_function %in% c("h", "halt", "halton")) {
            random_weights <- (rng_pars$max - rng_pars$min) * halton(nr_rows, N_hidden[1], init = TRUE) + rng_pars$min
        }
        else if (rng_function %in% c("s", "sobo", "sobol")) {
            random_weights <- (rng_pars$max - rng_pars$min) * sobol(nr_rows, N_hidden[1], init = TRUE) + rng_pars$min
        }
    }
    else {
        rng_pars$n <- N_hidden[1] * nr_rows
        random_weights <- matrix(do.call(rng_function, rng_pars), ncol = N_hidden[1]) 
    }
    
    W_hidden <- list(random_weights)
    
    ## Values of last hidden layer (before pre-training)
    H_tilde <- rwnn_forward(X = X, W = W_hidden, activation = activation, bias = bias_hidden)
    H_tilde <- lapply(seq_along(H_tilde), function(i) matrix(H_tilde[[i]], ncol = N_hidden[i]))
    H_tilde <- do.call("cbind", H_tilde)
    
    X_tilde <- X
    if (bias_hidden) {
        X_tilde <- cbind(1, X_tilde)
    }
    
    ## Auto-encoder pre-training
    if (method == "l1") {
        W_tilde <- lasso_ls(H_tilde, X_tilde, tau = 1, max_iterations = 1000, step_shrink = 0.001)$W
        W_hidden[[1]] <- t(W_tilde)
    } else if (method == "l2") {
        HT_tilde <- t(H_tilde)
        I_tilde <- diag(ncol(H_tilde))
        
        W_tilde <- solve(HT_tilde %*% H_tilde + I_tilde) %*% HT_tilde %*% X_tilde
        W_hidden[[1]] <- t(W_tilde)
    } else {
        stop("Method not implemented; set method to either \"l1\" or \"l2\".")
    }
    
    ## Values of last hidden layer (after pre-training)
    H <- rwnn_forward(X, W_hidden, activation, bias_hidden)
    H <- lapply(seq_along(H), function(i) matrix(H[[i]], ncol = N_hidden[i]))
    
    if (control$combine_hidden){
        H <- do.call("cbind", H)
    }
    else {
        H <- H[[length(H)]]
    }
    
    ## Estimate parameters in output layer
    if (control$bias_output) {
        H <- cbind(1, H)
    }
    
    O <- H
    if (control$combine_input) {
        O <- cbind(X, H)
    }
    
    W_output <- estimate_output_weights(O, y, control$lnorm, lambda)
    
    ## Return object
    object <- list(
        formula = NULL,
        data = if(control$include_data) list(X = X, y = y, C = colnames(y)) else NULL, 
        N_hidden = N_hidden, 
        activation = activation, 
        lambda = lambda,
        Bias = list(Hidden = bias_hidden, Output = control$bias_output),
        Weights = list(Hidden = W_hidden, Output = W_output$beta),
        Sigma = list(Hidden = NA, Output = W_output$sigma),
        Type = type,
        Combined = list(Input = control$combine_input, Hidden = control$combine_hidden)
    )
    
    class(object) <- "RWNN"
    return(object)
}


#' @rdname ae_rwnn
#' @method ae_rwnn formula
#' 
#' @example inst/examples/aerwnn_example.R
#' 
#' @export
ae_rwnn.formula <- function(formula, data = NULL, N_hidden = c(), lambda = NULL, method = "l1", type = NULL, control = list()) {
    # Checks for 'N_hidden'
    if (length(N_hidden) < 1) {
        stop("When the number of hidden-layers is 0, or left 'NULL', the RWNN reduces to a linear model, see ?lm.")
    }
    
    if (!is.numeric(N_hidden)) {
        stop("Not all elements of the 'N_hidden' vector were numeric.")
    }
    
    # Checks for 'data'
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
    
    # Checks for 'method'
    method <- tolower(method)
    if (!(method %in% c("l1", "l2"))) {
        stop("'method' has to be set to 'l1' or 'l2'.")
    }

    # Re-capture feature names when '.' is used in formula interface
    formula <- terms(formula, data = data)
    formula <- strip_terms(formula)
    
    #
    X <- model.matrix(formula, data)
    keep <- which(colnames(X) != "(Intercept)")
    if (any(colnames(X) == "(Intercept)")) {
        X <- X[, keep, drop = FALSE]
    }
    
    #
    y <- model.response(model.frame(formula, data))
    if (is.null(type)) {
        if (class(y) == "numeric") {
            type <- "regression"
            
            if (all(abs(y - round(y)) < 1e-8)) {
                warning("The response consists of only integers, is this a classification problem?")
            }
        }
        else if (class(y) %in% c("factor", "character", "logical")) {
            type <- "classification"
        }
    }
    
    y <- as.matrix(y, nrow = nrow(data))
    
    # Change output based on 'type'
    if (tolower(type) %in% c("c", "class", "classification")) {
        type <- "classification"
        
        y_names <- sort(unique(y))
        y <- factor(y, levels = y_names)
        y <- model.matrix(~ 0 + y)
        
        attr(y, "assign") <- NULL
        attr(y, "contrasts") <- NULL
        
        y <- 2 * y - 1
        
        colnames(y) <- paste(y_names, sep = "")
    } 
    else if (tolower(type) %in% c("r", "reg", "regression")) {
        type <- "regression"
    }
    else {
        stop("'type' has not been correctly specified, it needs to be set to either 'regression' or 'classification'.")
    }
    
    #
    mm <- ae_rwnn.matrix(X, y, N_hidden = N_hidden, lambda = lambda, method = method, type = type, control = control)
    mm$formula <- formula
    return(mm)
}