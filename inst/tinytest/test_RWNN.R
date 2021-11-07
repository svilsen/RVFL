#############################
#       Setting up data     #
#############################
set.seed(1)
N <- 2000
p <- 5

s <- seq(0, pi, length.out = N)
X <- matrix(NA, ncol = p, nrow = N)
X[, 1] <- sin(s)
X[, 2] <- cos(s)
X[, 3] <- s
X[, 4] <- s^2
X[, 5] <- s^3

beta <- matrix(rnorm(p), ncol = 1) 
y <- X %*% beta + rnorm(N, 0, 1)

N_hidden <- 20
lambda <- 0.1

########################################
#       Training models to compare     #
########################################
set.seed(1)
m_rwnn <- rwnn(X, y, N_hidden, lambda)

set.seed(1)
m_elm <- elm(X, y, N_hidden, lambda)

set.seed(1)
m_stackrwnn <- stack_rwnn(X, y, N_hidden, lambda, B = 10, optimise = TRUE, folds = 10)

set.seed(1)
m_bagrwnn <- bag_rwnn(X, y, N_hidden, lambda, B = 10)

set.seed(1)
m_samplerwnn <- sample_rwnn(X, y, N_hidden, lambda, control = list(method = "post"))

set.seed(1)
th <- tune_hyperparameters(rwnn, X, y, folds = 10, hyperparameters = list(lambda = c(0.1, 1, 2), N_hidden = list(c(N_hidden), c(N_hidden, N_hidden))))

######################################
#       Examining model results      #
######################################
#
expect_silent(m_rwnn)

expect_silent(m_elm)

expect_equal(m_rwnn$Weights$Output, m_elm$Weights$Output, tolerance = 1e-7)

expect_equal(round(mse(m_rwnn, X, y), 4), 1.076, tolerance = 1e-7)

#
expect_equal(round(m_bagrwnn$weights[1], 4), 0.1, tolerance = 1e-7)

expect_equal(round(mse(m_bagrwnn, X, y), 4), 1.1722, tolerance = 1e-7)

#
expect_equal(round(m_stackrwnn$weights[1], 4), 0.199, tolerance = 1e-7)

expect_equal(round(mse(m_stackrwnn, X, y), 4), 1.0776, tolerance = 1e-7)

# 
expect_equal(th$lambda, 2, tolerance = 1e-7)

expect_equal(th$N_hidden, 20, tolerance = 1e-7)

expect_equal(round(mse(th, X, y), 4), 1.0761, tolerance = 1e-7)

#
expect_equal(round(mse(m_samplerwnn, X, y), 4), 1.076, tolerance = 1e-7)


#################################################
#       Warnings/errors when casting data       #
#################################################
Z <- as.data.frame(X)
v <- as.data.frame(y)

expect_warning(rwnn(Z, y, N_hidden, lambda), "'X' has to be a matrix... trying to cast 'X' as a matrix.")

expect_warning(rwnn(X, v, N_hidden, lambda), "'y' has to be a matrix... trying to cast 'y' as a matrix.")

expect_warning(rwnn(X, cbind(y, v), N_hidden, lambda), "More than a single column was detected in 'y', only the first column is used in the model.")

expect_error(rwnn(X[1:10, ], y, N_hidden, lambda), "The number of rows in 'y' and 'X' do not match.")

#############################################################
#       Warnings/errors when not specifying paramters       #
#############################################################

#
expect_error(rwnn(X, y, NULL, lambda), "When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see \\?lm.")

expect_error(rwnn(X, y, c(), lambda), "When the number of hidden layers is 0, or left 'NULL', the RWNN reduces to a linear model, see \\?lm.")

expect_error(rwnn(X, y, c("10", 10), lambda), "Not all elements of the 'N_hidden' vector were numeric.")

#
expect_warning(rwnn(X, y, N_hidden, NULL), "Note: 'lambda' was not supplied/not numeric and set to 0.")

expect_warning(rwnn(X, y, N_hidden, "NULL"), "Note: 'lambda' was not supplied/not numeric and set to 0.")

expect_warning(rwnn(X, y, N_hidden, -1), "'lambda' has to be a real number larger than or equal to 0.")

expect_warning(rwnn(X, y, N_hidden, c(1, 1)), "The length of 'lambda' was larger than 1, only the first element will be used.")

#
expect_warning(rwnn(X, y, N_hidden, lambda, control = list(N_features = c(1, 1))), "The length of 'N_features' was larger than 1, only the first element will be used.")

expect_error(rwnn(X, y, N_hidden, lambda, control = list(N_features = 0)), "'N_features' have to be between 1 and the total number of features.")

#
expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, lnorm = "l3")), "'lnorm' has to be either 'l1' or 'l2'.")

#
expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, bias_hidden = c(T, F))), "The 'bias_hidden' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")

#
expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, activation = c("sigmoid", "linear"))), "The 'activation' vector specified in the control-object should have length 1, or be the same length as the vector 'N_hidden'.")

expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, activation = c("sii"))), "Invalid activation function detected in 'activation' vector. The implemented activation functions are: 'sigmoid', 'tanh', 'relu', 'silu', 'softplus', 'softsign', 'sqnl', 'gaussian', 'sqrbf', 'bentidentity', and 'identity'.")

#
expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, bias_output = "")), "'bias_output' has to be 'TRUE'/'FALSE'.")

#
expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, combine_input = "")), "'combine_input' has to be 'TRUE'/'FALSE'.")

#
expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, include_data = "")), "'include_data' has to be 'TRUE'/'FALSE'.")

#
expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, rng_pars = list())), "The following arguments were not found in 'rng_pars' list: min, max")

expect_error(do.call("control_rwnn", list(N_hidden = N_hidden, rng_pars = list(min = -1))), "The following arguments were not found in 'rng_pars' list: max")


################################################################################
#       Warnings/errors when not specifying paramters in RWNN extensions       #
################################################################################

#
expect_warning(ae_rwnn(X, y, c(10, 10), lambda), "More than one hidden was found, but this method is designed with a single hidden layer in mind, therefore, only the first element 'N_hidden' is used.")

expect_error(ae_rwnn(X, y, N_hidden, lambda, method = "l3"), "Method not implemented, please set method to either \"l1\" or \"l2\".")

#
expect_warning(bag_rwnn(X, y, N_hidden, lambda, B = NULL), "Note: 'B' was not supplied, 'B' was set to 100.")

#
expect_warning(boost_rwnn(X, y, N_hidden, lambda, B = NULL, epsilon = 0.1), "Note: 'B' was not supplied, 'B' was set to 10.")

expect_warning(boost_rwnn(X, y, N_hidden, lambda, B = 10, epsilon = NULL), "Note: 'epsilon' was not supplied and set to 1.")

expect_warning(boost_rwnn(X, y, N_hidden, lambda, B = 10, epsilon = "NULL"), "Note: 'epsilon' was not supplied and set to 1.")

expect_warning(boost_rwnn(X, y, N_hidden, lambda, B = 10, epsilon = -1), "'epsilon' has to be a number between 0 and 1.")

expect_warning(boost_rwnn(X, y, N_hidden, lambda, B = 10, epsilon = 2), "'epsilon' has to be a number between 0 and 1.")

#
expect_error(stack_rwnn(X, y, N_hidden, lambda, B = 100, optimise = "", folds = NULL), "'optimise' has to be 'TRUE'/'FALSE'.")

expect_warning(stack_rwnn(X, y, N_hidden, lambda, B = 100, optimise = TRUE, folds = NULL), "Note: 'folds' was not supplied, and is set to 10.")

expect_warning(stack_rwnn(X, y, N_hidden, lambda, B = NULL, optimise = FALSE, folds = 10), "Note: 'B' was not supplied, 'B' was set to 100.")


#
expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, method = "bbbbbb")), "The argument supplied to 'method' is not implemented, please set method to 'map', 'stack', or 'posterior'")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, N_simulations = NULL)), "'N_simulations' has to be numeric.")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, N_simulations = "")), "'N_simulations' has to be numeric.")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, N_simulations = 0)), "'N_simulations' has to be larger than 0.")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, N_burnin = NULL)), "'N_burnin' has to be numeric.")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, N_burnin = "")), "'N_burnin' has to be numeric.")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, N_simulations = 1, N_burnin = 2)), "'N_burnin' has be smaller than 'N_simulations'.")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, method = "stack", N_resample = NULL)), "'N_resample' has to be numeric.")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, method = "stack", N_resample = "")), "'N_resample' has to be numeric.")

expect_error(do.call("control_sample_rwnn", list(N_hidden = N_hidden, method = "stack", N_resample = 10000)), "'N_resample' has be smaller than 'N_simulations - N_burnin'.")


###########################################################
#       Warnings/errors when tuning hyperparameters       #
###########################################################

#
expect_error(tune_hyperparameters(""), "'method' has to be a function.")

expect_error(tune_hyperparameters(lm), "The tuning function is only implemented for 'RWNN' and 'ERWNN' methods.")

expect_warning(tune_hyperparameters(sample_rwnn), "Support for 'sample_rwnn' is not implemented.")

#
expect_error(tune_hyperparameters(rwnn, X, y, folds = NULL, hyperparameters = list(N_hidden = list(N_hidden), lambda = lambda)), "'folds' was either 'NULL' or not numeric.")

expect_error(tune_hyperparameters(rwnn, X, y, folds = "NULL", hyperparameters = list(N_hidden = list(N_hidden), lambda = lambda)), "'folds' was either 'NULL' or not numeric.")

expect_warning(tune_hyperparameters(rwnn, X, y, folds = 0, hyperparameters = list(N_hidden = list(N_hidden), lambda = lambda)), "'folds' was smaller than 1, setting 'folds' to 3.")

expect_warning(tune_hyperparameters(rwnn, X, y, folds = 1, hyperparameters = list(N_hidden = list(N_hidden), lambda = lambda)), "'folds' was equal to 1, this is not recommended, as no validation set is generated.")

expect_warning(tune_hyperparameters(rwnn, X, y, folds = 2 * N, hyperparameters = list(N_hidden = list(N_hidden), lambda = lambda)), "'folds' was larger than the number of observations, setting 'folds' equal to the number of observations.")

#
expect_warning(tune_hyperparameters(rwnn, X, y, folds = 10, hyperparameters = list(N_hidden = list(N_hidden), lambda = lambda), trace = NULL), "'trace' was either 'NULL' or not numeric, setting 'trace' to 0")

expect_warning(tune_hyperparameters(rwnn, X, y, folds = 10, hyperparameters = list(N_hidden = list(N_hidden), lambda = lambda), trace = "NULL"), "'trace' was either 'NULL' or not numeric, setting 'trace' to 0")

#
expect_error(tune_hyperparameters(rwnn, X, y, folds = 10, hyperparameters = list(N_hidden = list(N_hidden)), trace = 0), "Missing arguments in 'hyperparameters': lambda")

expect_error(tune_hyperparameters(rwnn, X, y, folds = 10, hyperparameters = list(), trace = 0), "Missing arguments in 'hyperparameters': N_hidden, lambda")


#############################################################
#       Warnings/errors when using RWNN AUX functions       #
#############################################################

#
expect_equal(m_rwnn$Weights$Output, coef(m_rwnn), tolerance = 1e-7)

#
expect_error(predict(rwnn(X, y, N_hidden, lambda, control = list(include_data = FALSE))), "The RWNN-object does not contain any data: Either supply 'newdata', or re-create object with 'include_data = TRUE' \\(default\\).")

expect_error(predict(m_rwnn, newdata = X[, -1]), "The number of features \\(columns\\) provided in 'newdata' does not match the number of features of the model.")

##############################################################
#       Warnings/errors when using ERWNN AUX functions       #
##############################################################

#
expect_equal(round(coef(m_bagrwnn, type = "m"), 4)[1], -1.5747, tolerance = 1e-7)

expect_error(coef(m_bagrwnn, type = "hhh"), "The value of 'type' was not valid, see '\\?coef.ERWNN' for valid options of 'type'.")

#
expect_equal(round(predict(m_bagrwnn, newdata = matrix(X[1, ], nrow = 1), type = "a"), 4)[1, 1], 0.8135, tolerance = 1e-7)

expect_equal(round(predict(m_bagrwnn, newdata = matrix(X[1, ], nrow = 1), type = "m"), 4)[1, ], -0.4277, tolerance = 1e-7)

expect_equal(round(predict(m_bagrwnn, newdata = matrix(X[1, ], nrow = 1), type = "sd"), 4)[1, ], 2.0311, tolerance = 1e-7)

expect_error(predict(m_bagrwnn, type = "hhh"), "The value of 'type' was not valid, see '\\?predict.ERWNN' for valid options of 'type'.")

#
expect_equal(set_weights(m_bagrwnn, rep(0.1, 10))$weights[1], 0.1, tolerance = 1e-7)

expect_warning(set_weights(m_bagrwnn, NULL), "No weights defined, setting weights to uniform.")

expect_error(set_weights(m_bagrwnn, rep(0.1, 9)), "The number of supplied weights have to be equal to the number of ensemble weights.")

expect_error(set_weights(m_bagrwnn, rep(1.1, 10)), "The weights have to sum to 1.")

expect_error(set_weights(m_bagrwnn, c(-0.1, 1.1, rep(0, 8))), "All weights have to be between 0 and 1.")

#
expect_equal(round(estimate_weights(m_bagrwnn, X_val = X, y_val = y)$weights, 4)[1], 0.4916, tolerance = 1e-7)

expect_warning(estimate_weights(m_bagrwnn), "The validation-set was not properly specified, therefore, the training-set is used for weight estimation.")


##############################################################
#       Warnings/errors when using SRWNN AUX functions       #
##############################################################

#
expect_equal(round(coef(m_samplerwnn, parameter = "w")[[1]][[1]][1, 1], 4), -0.3708, tolerance = 1e-7)

expect_equal(round(coef(m_samplerwnn, parameter = "beta")[[1]][1, 1], 4), 0.3445, tolerance = 1e-7)

expect_equal(round(coef(m_samplerwnn, parameter = "sigma")[[1]], 4), 1.0428, tolerance = 1e-7)

expect_error(coef(m_samplerwnn, parameter = "m"), "The value of 'parameter' was not valid, see '\\?coef.SRWNN' for valid options of 'parameter'.")

#
expect_equal(round(predict(m_samplerwnn, newdata = matrix(X[1, ], nrow = 1), type = "a"), 4)[1, 1], 0.2863, tolerance = 1e-7)

expect_equal(round(predict(m_samplerwnn, newdata = matrix(X[1, ], nrow = 1), type = "m"), 4)[1, ], 0.2876, tolerance = 1e-7)

expect_equal(unname(round(predict(m_samplerwnn, newdata = matrix(X[1, ], nrow = 1), type = "ci"), 4)[1, ]), c(0.2856, 0.2893), tolerance = 1e-7)

expect_error(predict(m_samplerwnn, type = "hhh"), "The value of 'type' was not valid, see '\\?predict.SRWNN' for valid options of 'type'.")


