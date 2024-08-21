#############################
#       Setting up data     #
#############################
set.seed(1)
N <- 2000

s <- runif(N, -5, 5)
t <- runif(N, 0, 2)
u <- rnorm(N, 0, 2)
x <- rgamma(N, 2, 4)
b <- rbeta(N, 10, 4)

X <- cbind(1, s, t, u, x, b)
beta <- c(-1, 2, 1, 2, 0.5, 3)

y <- 1 / (1 + exp(-X%*%beta + rnorm(N, 0, 0.1)))
data <- data.frame("y" = y, "X1" = s, "X2" = t, "X3" = u, "X4" = x, "X5" = b)

n_hidden <- 20
lambda <- 0.1

mse <- function(m, X, y) {
    return(mean((y - predict(m, newdata = X))^2))
}

########################################
#       Training models to compare     #
########################################
set.seed(1)
m_rwnn <- rwnn(y ~ ., data, n_hidden = n_hidden, lambda = lambda) 

set.seed(1)
m_aerwnn <- ae_rwnn(y ~ ., data, n_hidden = n_hidden, lambda = lambda, method = "l2")

set.seed(1)
m_edrwnn <- ed_rwnn(y ~ ., data, n_hidden = n_hidden, lambda = lambda)

set.seed(1)
m_stackrwnn <- stack_rwnn(y ~ ., data, n_hidden = n_hidden, lambda = lambda, B = 10, optimise = TRUE, folds = 10)

set.seed(1)
m_bagrwnn <- bag_rwnn(y ~ ., data, n_hidden = n_hidden, lambda = lambda, B = 10)

set.seed(1)
m_boostrwnn <- boost_rwnn(y ~ ., data, n_hidden = n_hidden, lambda = lambda, B = 10, epsilon = 0.01)

######################################
#       Examining model results      #
######################################
# RWNN
expect_silent(m_rwnn)
expect_equal(round(mse(m_rwnn, data, y), 4), 0.0421, tolerance = 1e-7)

# 
expect_silent(m_aerwnn)
expect_equal(round(mse(m_aerwnn, data, y), 4), 0.0421, tolerance = 1e-7)

#
expect_equal(round(m_edrwnn$weights[1], 4), 1, tolerance = 1e-7)
expect_equal(round(mse(m_edrwnn, data, y), 4), 0.0421, tolerance = 1e-7)

#
expect_equal(round(m_stackrwnn$weights[1], 4), 0.0212, tolerance = 1e-7)
expect_equal(round(mse(m_stackrwnn, data, y), 4), 0.0434, tolerance = 1e-7)

#
expect_equal(round(m_bagrwnn$weights[1], 4), 0.1, tolerance = 1e-7)
expect_equal(round(mse(m_bagrwnn, data, y), 4), 0.1262, tolerance = 1e-7)

#
expect_equal(round(m_boostrwnn$weights[1], 4), 0.01, tolerance = 1e-7)
expect_equal(round(mse(m_boostrwnn, data, y), 4), 0.4651, tolerance = 1e-7)

#############################################################
#       Warnings/errors when not specifying parameters      #
#############################################################
## RWNN
#
expect_error(rwnn(y ~ ., data = data, NULL, lambda), "When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see \\?lm.")
expect_error(rwnn(y ~ ., data = data, c(), lambda), "When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see \\?lm.")

expect_error(rwnn(y ~ ., data = data, c("10", 10), lambda), "Not all elements of the 'n_hidden' vector were numeric.")


#
expect_error(rwnn(z ~ e, data = NULL, n_hidden = n_hidden, lambda = 0.1), "'data' needs to be supplied when using 'formula'.")

#
expect_warning(rwnn(I(round(y)) ~ ., data = data, n_hidden = n_hidden, lambda = 0.1), "The response consists of only integers, is this a classification problem?")
expect_error(rwnn(y ~ ., data = data, n_hidden = n_hidden, lambda = 0.1, type = "a"), "'type' has not been correctly specified, it needs to be set to either 'regression' or 'classification'.")


#
expect_warning(rwnn(y ~ ., data = data, n_hidden, NULL), "Note: 'lambda' was not supplied, or not numeric, and is therefore set to 0.")
expect_warning(rwnn(y ~ ., data = data, n_hidden, "NULL"), "Note: 'lambda' was not supplied, or not numeric, and is therefore set to 0.")
expect_warning(rwnn(y ~ ., data = data, n_hidden, -1), "'lambda' has to be a real number larger than or equal to 0.")
expect_warning(rwnn(y ~ ., data = data, n_hidden, c(1, 1)), "The length of 'lambda' was larger than 1, only the first element will be used.")

#
expect_warning(rwnn(y ~ ., data = data, n_hidden, lambda, control = list(n_features = c(1, 1))), "The length of 'n_features' was larger than 1, only the first element will be used.")
expect_error(rwnn(y ~ ., data = data, n_hidden, lambda, control = list(n_features = 0)), "'n_features' has to be between 1 and the total number of features.")

## Control
#
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, lnorm = "l3")), "'lnorm' has to be either 'l1' or 'l2'.")

#
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, bias_hidden = c(T, F))), "The 'bias_hidden' vector specified in the control-object should have length 1, or be the same length as the vector 'n_hidden'.")
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, bias_output = c("a"))), "'bias_output' has to be 'TRUE'/'FALSE'.")

#
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, combine_input = "")), "'combine_input' has to be 'TRUE'/'FALSE'.")
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, combine_hidden = "")), "'combine_hidden' has to be 'TRUE'/'FALSE'.")

#
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, include_data = "")), "'include_data' has to be 'TRUE'/'FALSE'.")
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, include_estimate = "")), "'include_estimate' has to be 'TRUE'/'FALSE'.")

#
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, activation = c("sigmoid", "linear"))), "The 'activation' vector specified in the control-object should have length 1, or be the same length as the vector 'n_hidden'.")
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, activation = c("sii"))), "Invalid activation function detected in 'activation' vector. The implemented activation functions are: 'sigmoid', 'tanh', 'relu', 'silu', 'softplus', 'softsign', 'sqnl', 'gaussian', 'sqrbf', 'bentidentity', and 'identity'.")

#
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, rng = "a")), "object 'a' of mode 'function' was not found")
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, rng_pars = list())), "The following arguments were not found in 'rng_pars' list: min, max")
expect_error(do.call("control_rwnn", list(n_hidden = n_hidden, rng_pars = list(min = -1))), "The following arguments were not found in 'rng_pars' list: max")


#################################################################################
#       Warnings/errors when not specifying parameters in RWNN extensions       #
#################################################################################

## AE
expect_error(ae_rwnn(y ~ ., data = data, n_hidden, lambda, method = "l3"), "'method' has to be set to 'l1' or 'l2'.")
expect_warning(ae_rwnn(y ~ ., data = data, n_hidden, lambda = c(1, 2, 3), method = "l2"), "The length of 'lambda' was larger than 2; only the first two elements will be used.")

## ED
expect_warning(ed_rwnn(y ~ ., data = data, n_hidden, lambda = lambda, control = list(combine_hidden = TRUE)), "'combine_hidden' has to be set to 'FALSE' for the 'ed_rwnn' model to function correctly.")

## Bagging
expect_warning(bag_rwnn(y ~ ., data = data, n_hidden, lambda, B = NULL), "Note: 'B' was not supplied and is therefore set to 100.")

## Boosting
#
expect_warning(boost_rwnn(y ~ ., data = data, n_hidden, lambda, B = NULL, epsilon = 0.1), "Note: 'B' was not supplied and is therefore set to 100.")

#
expect_warning(boost_rwnn(y ~ ., data = data, n_hidden, lambda, B = 10, epsilon = NULL), "Note: 'epsilon' was not supplied and is therefore set to 0.1")
expect_warning(boost_rwnn(y ~ ., data = data, n_hidden, lambda, B = 10, epsilon = "NULL"), "Note: 'epsilon' was not supplied and is therefore set to 0.1.")
expect_warning(boost_rwnn(y ~ ., data = data, n_hidden, lambda, B = 10, epsilon = -1), "'epsilon' has to be a number between '0' and '1'.")
expect_warning(boost_rwnn(y ~ ., data = data, n_hidden, lambda, B = 10, epsilon = 2), "'epsilon' has to be a number between '0' and '1'.")

## Stacking
expect_error(stack_rwnn(y ~ ., data = data, n_hidden, lambda, B = 100, optimise = "", folds = NULL), "'optimise' has to be 'TRUE'/'FALSE'.")

expect_warning(stack_rwnn(y ~ ., data = data, n_hidden, lambda, B = 100, optimise = TRUE, folds = NULL), "Note: 'folds' was not supplied and is therefore set to 10.")

expect_warning(stack_rwnn(y ~ ., data = data, n_hidden, lambda, B = NULL, optimise = FALSE, folds = 10), "Note: 'B' was not supplied and is therefore set to 100.")


########################################################
#       Warnings/errors when using AUX functions       #
########################################################

## RWNN
expect_error(predict(rwnn(y ~ ., data = data, n_hidden, lambda, control = list(include_data = FALSE))), "The RWNN-object does not contain any data. Use the 'newdata' argument, or re-create 'RWNN-object' setting 'include_data = TRUE' \\(default\\).")
expect_error(predict(m_rwnn, newdata = data[, -2]), "object 'X1' not found")

## ERWNN
#
expect_equal(round(predict(m_bagrwnn, newdata = data[1, , drop = FALSE], type = "a")[[1]], 4)[1, 1], 0.4361, tolerance = 1e-7)
expect_equal(round(predict(m_bagrwnn, newdata = as.data.frame(data[1, , drop = FALSE]), type = "m"), 4)[1, ], 0.5016, tolerance = 1e-7)
expect_equal(round(predict(m_bagrwnn, newdata = as.data.frame(data[1, , drop = FALSE]), type = "std"), 4)[1, ], 0.0234, tolerance = 1e-7)

expect_error(predict(m_bagrwnn, type = "hhh"), "The value of 'type' was not valid, see '\\?predict.ERWNN' for valid options of 'type'.")

#
m_bagrwnn_cl <- m_bagrwnn
m_bagrwnn_cl$models[[1]]$type <- "classification"

expect_warning(predict(m_bagrwnn_cl), "Multiple 'type' fields found among the ensemble models; therefore, only the first ensemble model is used to determine model type.")

## Classify
expect_error(classify(matrix(c(1, 2, 3, 4), ncol = 1), C = matrix(0, ncol = 2, nrow = 4)), "The number of columns 'y' has to match the number of elements in 'C'.")

##############################################################
#       Warnings/errors when using reduction functions       #
##############################################################

## 
expect_warning(reduce_network(m_rwnn, "mag", retrain = ""), "'retrain' is set to 'TRUE' as it was either 'NULL', or not 'logical'.")

expect_error(reduce_network(rwnn(y ~ ., data = data, n_hidden, lambda, control = list(include_data = FALSE)), "mag", retrain = TRUE), "Data has to be present in the model object, or supplied through '...' argument as 'X = ' and 'y = '.")

expect_error(reduce_network(m_rwnn, "", retrain = TRUE), "'method' is either not implemented, or not a function.")

## 
# MAG / UNIF / LAMP
expect_warning(reduce_network(m_rwnn, "mag", retrain = TRUE), "'p' is set to '0.1' as it was either 'NULL', or not 'numeric'.")

expect_warning(reduce_network(m_rwnn, "unif", retrain = TRUE, p = -1), "'p' is set to '0.01', because it was found to be smaller than '0'.")

expect_warning(reduce_network(m_rwnn, "lamp", retrain = TRUE, p = 2), "'p' is set to '0.99', because it was found to be larger than '1'.")

# APoZ / L2
expect_warning(reduce_network(m_rwnn, "apoz", retrain = TRUE), "APOZ was designed for 'relu' activation functions, but no 'relu' activation was found.")

expect_error(reduce_network(rwnn(y ~ ., data = data, n_hidden, lambda, control = list(activation = "relu")), "apoz", retrain = TRUE, type = ""), "'type' should be either 'global' or 'uniform'.")

expect_warning(reduce_network(rwnn(y ~ ., data = data, n_hidden, lambda, control = list(activation = "relu")), "apoz", retrain = TRUE, tolerance = NULL), "'tolerance' is set to '1e-8' as it was either 'NULL', or not 'numeric'.")

expect_warning(reduce_network(rwnn(y ~ ., data = data, n_hidden, lambda, control = list(activation = "relu")), "apoz", retrain = TRUE, tolerance = -2), "'tolerance' is set to '1e-8', because it was found to be smaller than '0'.")


# COR / CORTEST
expect_warning(reduce_network(m_rwnn, "cor", retrain = TRUE), "'rho' is set to '0.99' as it was either 'NULL', or not 'numeric'.")

expect_warning(reduce_network(m_rwnn, "cor", retrain = TRUE, rho = -1), "'rho' is set to '0.01', because it was found to be smaller than '0'.")

expect_warning(reduce_network(m_rwnn, "cor", retrain = TRUE, rho = 2), "'rho' is set to '0.99', because it was found to be larger than '1'.")

expect_warning(reduce_network(m_rwnn, "cortest", retrain = TRUE, rho = 0.99), "'alpha' is set to '0.05' as it was either 'NULL', or not 'numeric'.")

expect_warning(reduce_network(m_rwnn, "cortest", retrain = TRUE, rho = 0.99, alpha = -1), "'alpha' is set to '0.01', because it was found to be smaller than '0'.")

expect_warning(reduce_network(m_rwnn, "cortest", retrain = TRUE, rho = 0.99, alpha = 2), "'alpha' is set to '0.99', because it was found to be larger than '1'.")

#
expect_error(reduce_network(m_rwnn, "relief", retrain = TRUE, type = ""), "'type' should be either 'weight' or 'neuron'.")

# 
expect_error(reduce_network(m_bagrwnn, method = "stack", tolerance = 1e-8), "Setting 'method' to 'stacking' is only meant for stacking ensemble models.")
expect_error(reduce_network(m_stackrwnn, method = "stack", tolerance = 3), "Because of the chosen tolerance all models were removed; the tolerance should be lowered to a more appropriate level.")

