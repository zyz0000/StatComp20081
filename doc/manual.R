## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
library(StatComp20081)
X <- array(1:24, dim = c(2, 3, 4))
VAR(X)

## -----------------------------------------------------------------------------
PROJ(X)

## -----------------------------------------------------------------------------
A.init <- matrix(1:4, 2, 2)
B.init <- matrix(1:9, 3, 3)
max.iters <- 200
ILS(X, A.init, B.init, max.iters)

## -----------------------------------------------------------------------------
A.init <- matrix(1:4, 2, 2)
B.init <- matrix(1:9, 3, 3)
sigmar.init <- diag(3)
sigmac.init <- diag(2)
max.iters <- 200
MLE(A.init, B.init, sigmac.init, sigmar.init, X, max.iters)

