# Random weight neural networks in R: The RWNN package
The `RWNN`-package implements a variety of random weight neural networks (RWNNs), a simplification of feed-forward neural networks,  `R` using `Rcpp` and `RcppArmadillo`. RWNNs randomly assigns the weights of the network between the input-layer and the last hidden-layer, focusing on training the weights between the last hidden-layer and the output-layer. This simplification makes training an RWNN as fast, simple, and efficient as training a (regularised) linear model. However, by randomly drawing the weights between the input and last hidden-layer additional instability is introduced into the network. Therefore, various extensions including deep RWNNs, sparse RWNNs, and ensemble deep RWNNs have been proposed to combat this instability.

The `RWNN` package allows the user to create an RWNN of any depth, letting the user set the number of neurons and activation functions used in each layer, choose the sampling distribution of the random weights, and during estimation of the output-weights it allows for Moore-Penrose inversion, $\ell_1$ regularisation, and $\ell_2$ regularisation.

Along with a standard RWNN implementation, a number of popular variations have also been implemented in the `RWNN` package: 

-   **ELM** (extreme learning machine): A simplified version of an RWNN without a link between the input and output layer (this is also the default behaviour of the RWNN implementation).
-   **deep RWNN**: An RWNN with multiple hidden layers, where the output of each hidden-layer is included as features in the model.
-   **sparse RWNN**: Applies sparse auto-encoder ($\ell_1$ regularised) pre-training to reduce the number non-zero weights between the input and the hidden layer (the implementation generalises this concept to allow for both $\ell_1$ and $\ell_2$ regularisation).
-   **ensemble deep RWNN**: An ensemble extension of deep RWNNs using the output of each hidden layer to create an ensemble of RWNNs, each hidden-layer being used to create a separate prediction of the target.

Furthermore, the `RWNN` package also includes general implementations of the following ensemble methods (using RWNNs as base learners):

-   **Stacking**: Stack multiple randomly generated RWNNs, deep RWNNs, or sparse RWNNs and estimate their contribution to a weighted ensemble using $k$-fold cross-validation.
-   **Bagging**: Bootstrap aggregation of RWNNs, deep RWNNs, or sparse RWNNs creates a number of bootstrap samples, sampled with replacement from the training-set. Furthermore, as in random forest, instead of using all features when training each RWNN, a subset of the features are chosen at random.
-   **Boosting**: The boosting implementation is based on stochastic gradient boosting using RWNN's, deep RWNN's, or sparse RWNN's as the base learner. 

Lastly, in order to improve computational time and memory efficiency, the `RWNN` package includes the following methods for pruning the number of weights and neurons: 

-   **Global magnitude** (weight pruning): Pruning a pre-defined proportion of the weights with lowest magnitude globally across the network.
-   **Uniform magnitude** (weight pruning): Pruning a pre-defined proportion of the weights with lowest magnitude layer-by-layer.
-   **LAMP** (weight pruning): Pruning a pre-defined proportion of the weights with lowest LAMP (layer-adaptive magnitude-based pruning) score globally across the network. The LAMP score is essentially a re-scaled magnitude.
-   **APoZ** (neuron pruning): Pruning a pre-defined proportion of neurons in layers, activated by the ReLU activation function, based on the average proportion of zeros (APoZ).
-   **Correlation** (neuron pruning): Pruning based on the pair-wise correlations between neurons layer-by-layer. A neuron is removed if its correlation exceeds a pre-defined threshold, or if its correlation is not significantly smaller than this pre-defined threshold.
-   **Relief** (weight and neuron pruning): Pruning a pre-defined proportion of the weights that, on average, provide only a small contribution to the next layer. 

# Installation

The `RWNN`-package depends on `R` (>= 4.1), `Rcpp` (>= 1.0.4.6), `RcppArmadillo`, `quadprog`, and `randtoolbox`. The package is not available on CRAN, therefore, `devtools` is needed to install the package from github. 

From R, run the following commands:  
```r
install.packages("Rcpp")
install.packages("RcppArmadillo")
install.packages("quadprog")
install.packages("randtoolbox")

install.packages("devtools")
devtools::install_github("svilsen/RWNN")
```

# Usage
In the following the data is randomly generated and split into training and validation sets. After which a few models are fitted, reduced, and compared.

```r
##
library("RWNN")

## Data set-up
data(example_data)

# Split data into training and validtion sets
tr <- sample(nrow(example_data), round(0.6 * nrow(example_data)))
example_train <- example_data[tr,]
example_val <- example_data[-tr,]

## Fitting models
n_hidden <- c(10, 15, 5)
lambda <- 0.01

# RWNN
m_rwnn <- rwnn(y ~ ., data = example_train, n_hidden = n_hidden, lambda = lambda)

# sp-RWNN
m_sprwnn <- ae_rwnn(y ~ ., data = example_train, n_hidden = n_hidden, lambda = c(lambda, 0.2), method = "l1")

# Reducing RWNN
m_rwnn_p <- m_rwnn |> 
    reduce_network(method = "correlation", rho = 0.9) |> 
    reduce_network(method = "lamp", p = 0.2) |> 
    reduce_network(method = "output")

## Ensemble methods    
# Bagging RWNN
m_bag <- bag_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda, B = 150)

# Boosting RWNN
m_boost <- boost_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda, B = 2000, epsilon = 0.005)

# Stacking RWNN
m_stack <- stack_rwnn(y ~ ., data = example_data, n_hidden = n_hidden, lambda = lambda, B = 25, optimise = TRUE)

# Removing RWNNs from stack with weights less than 1e-6 
m_stack_p <- m_stack |> 
    reduce_network(method = "stack", tolerance = 1e-6)

# ed-RWNN
m_ed <- ed_rwnn(y ~ ., data = example_train, n_hidden = n_hidden, lambda = lambda)

## Compare
rmse <- function(m, data_val) {
    return(sqrt(mean((data_val$y - predict(m, newdata = data_val))^2)))
}

(rmse_comp <- 
        data.frame(
            Method = c("RWNN", "sp-RWNN", "Pruning", "Bagging", "Boosting", "Stacking", "L-Stacking", "ed-RWNN"),
            RMSE = sapply(list(m_rwnn, m_sprwnn, m_rwnn_p, m_bag, m_boost, m_stack, m_stack_p, m_ed), rmse, data_val = example_data)
        )
)

```

Hyper-parameters can be optimised by the `caret` package using their generic interface. The following is an example optimising the hyper-parameters of a boosted RWNN. 
```r
library("caret")

lp_boostrwnn <- list(
    type = "Regression",
    library = "RWNN",
    parameters = list(
        parameter = c("n_hidden", "lambda1", "lambda2", "B", "epsilon"),
        class = c("numeric", "numeric", "numeric", "numeric", "numeric"),
        label = c("n_hidden", "lambda1", "lambda2", "B", "epsilon")
    ),
    grid = function(x, y, len = NULL, search = "grid") {
        if (search == "grid") {
            out <- expand.grid(
                n_hidden = 10 * seq_len(len),
                lambda1 = 10^seq(-1, -len), 
                lambda2 = 10^seq(-1, -len),
                B = round(10^seq(0, len - 1)),
                epsilon = 10^seq(-1, -len)
            )
        } 
        
        return(out)
    },
    fit = function(x, y, wts, param, lev, last, weights, classProbs, ...) {
        RWNN::boost_rwnn(
            y ~ x, 
            n_hidden = param$n_hidden,
            lambda = c(param$lambda1, param$lambda2), 
            B = param$B, 
            epsilon = param$epsilon, 
            method = "l1", 
            type = "regression",
            ...
        )
    },
    predict = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
        predict(object = modelFit, newdata = newdata) 
    }, 
    prob = NULL, 
    sort = NULL
)

tr <- sample(nrow(example_data), round(0.6 * nrow(example_data)))
example_train <- example_data[tr,]
example_val <- example_data[-tr,]

fit_control <- trainControl(
    method = "cv",
    number = 5,
    allowParallel = TRUE,
    verboseIter = TRUE
)

optimal_boost_rwnn <- train(
    y ~ ., 
    data = example_train, 
    method = lp_boostrwnn, 
    preProc = c("center", "scale"),
    tuneLength = 3,
    trControl = fit_control
)

plot(optimal_boost_rwnn)
```

## License

This project is licensed under the MIT License.
