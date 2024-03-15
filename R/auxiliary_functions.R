###################################################
####################### AUX #######################
###################################################

data_checks <- function(y, X) {
    if (!is.matrix(X)) {
        warning("'X' has to be a matrix... trying to cast 'X' as a matrix.")
        X <- as.matrix(X)
    }
    
    if (!is.matrix(y)) {
        warning("'y' has to be a matrix... trying to cast 'y' as a matrix.")
        y <- as.matrix(y)
    }
    
    if (dim(y)[1] != dim(X)[1]) {
        stop("The number of rows in 'y' and 'X' do not match.")
    }
    
    return(invisible(NULL))
}

strip_terms <- function(formula) {
    attr_names <- names(attributes(formula))
    for (i in seq_along(attr_names)) {
        attr(formula, attr_names[i]) <- NULL
    }
    
    formula <- as.formula(formula)
    return(formula)
}

orthonormal <- function(M) {
    # 
    svdM <- svd(M)
    U <- svdM$u
    S <- svdM$d
    
    #
    tol <- max(dim(M)) * max(S) * .Machine$double.eps
    R <- sum(S > tol)
    
    #
    X <- U[, 1:R, drop = FALSE]
    return(X)
}

random_orthonormal <- function(w, nr_rows, X, W_hidden, n_hidden, activation, bias_hidden) {
    W <- matrix(runif(n_hidden[w] * nr_rows), nrow = n_hidden[w])
    W <- orthonormal(W)
    
    if (nr_rows > n_hidden[w]) {
        if (w == 1) {
            Z <- X
        }
        else {
            Z <- rwnn_forward(X, W_hidden[seq_len(w - 1)], activation, bias_hidden)
            Z <- matrix(Z[[length(Z)]], ncol = n_hidden[w - 1])
        }
        
        if (bias_hidden[w]) {
            Z <- cbind(1, Z)
        }
        
        pca <- princomp(Z)
        L <- unname(t(pca$loadings[, seq_len(n_hidden[w]), drop = FALSE]))
        W <- W %*% L
    }
    
    W <- t(W)
    return(W)
}

#' Classifier
#' 
#' @description Function classifying an observation.
#' 
#' @param y A matrix of predicted classes.
#' @param C A vector of class names corresponding to the columns of \code{y}.
#' @param t The decision threshold which the predictions have to exceed (default is '0'). 
#' @param b A buffer which the largest prediction has to exceed when compared to the second largest prediction (default is '0').
#' 
#' @return A vector of class predictions.
#' 
#' @export 
classify <- function(y, C, t = NULL, b = NULL) {
    #
    if (dim(y)[2] != length(C)) {
        stop("The number of columns 'y' has to match the number of elements in 'C'.")
    }
    
    # 
    if (is.null(t)) {
        t <- 0.0
    }
    
    #
    if (is.null(b)) {
        b <- 0.0
    }
    
    #
    yc <- classify_cpp(y, C, t, b)
    return(yc)
}

