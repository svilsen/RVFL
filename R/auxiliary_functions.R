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

create_folds <- function(X, folds) {
    N <- nrow(X)
    index <- sample(N, N, replace = FALSE)
    fold_index <- rep(seq_len(folds), each = floor(N / folds))
    
    if (length(fold_index) < length(index)) {
        fold_index <- c(fold_index, seq_len(folds)[seq_len(length(index) - length(fold_index))])
    }
    
    return(unname(split(x = index, f = fold_index)))
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

random_orthonormal <- function(w, nr_rows, X, W_hidden, N_hidden, activation, bias_hidden) {
    W <- matrix(runif(N_hidden[w] * nr_rows), nrow = N_hidden[w])
    W <- orthonormal(W)
    
    if (nr_rows > N_hidden[w]) {
        if (w == 1) {
            Z <- X
        }
        else {
            Z <- rwnn_forward(X, W_hidden[seq_len(w - 1)], activation, bias_hidden)
            Z <- matrix(Z[[length(Z)]], ncol = N_hidden[w - 1])
        }
        
        if (bias_hidden[w]) {
            Z <- cbind(1, Z)
        }
        
        pca <- princomp(Z)
        L <- unname(t(pca$loadings[, seq_len(N_hidden[w]), drop = FALSE]))
        W <- W %*% L
    }
    
    W <- t(W)
    return(W)
}

estimate_weights_stack <- function(C, b, B) {
    # Creating matricies for QP optimisation problem.
    # NB: diagonal matrix is added to ensure the matrix is invertible due to potential numeric instability.
    D <- t(C) %*% C + diag(1e-8, nrow = ncol(C), ncol = ncol(C))
    d <- t(C) %*% b
    A <- rbind(t(matrix(rep(1, B), ncol = 1)), diag(B), -diag(B))
    b <- c(1, rep(0, B), rep(-1, B))
    
    # Solution to QP optimisation problem
    w <- solve.QP(D, d, t(A), b, meq = 1)$solution
    
    # Ensure all weights are valid (some may not be due to machine precision)
    w[w < 1e-16] <- 1e-16
    w[w > (1 - 1e-16)] <- (1 - 1e-16)
    w <- w / sum(w)
    
    return(w)
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

