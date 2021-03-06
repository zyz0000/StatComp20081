# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' @title The iterative least square estimator
#' @description The iterative least square estimator using Rcpp and RcppArmadillo
#' @param X a tensor of shape (m, n, t), where m is the number of rows, n is the number of columns, and t is the number of observations
#' @param A_init the initial value of A
#' @param B_init the initial value of B
#' @param max_iters the maximum iterations
#' @return a list containing the estimate of A and B
#' @examples
#' \dontrun{
#'   ils.est <- ILS(X, A_init, B_init, 200)
#' }
#' @export
ILS <- function(X, A_init, B_init, max_iters) {
    .Call(`_StatComp20081_ILS`, X, A_init, B_init, max_iters)
}

#' @title The maximum likelihood estimator
#' @description The maximum likelihood estimator using Rcpp and RcppArmadillo
#' @param A_init the initial value of A
#' @param B_init the initial value of B
#' @param sigmar_init the initial value of sigmar
#' @param sigmac_init the initial value of sigmac
#' @param X a tensor of shape (m, n, t), where m is the number of rows, n is the number of columns, and t is the number of observations
#' @param max_iters the maximum iterations
#' @return a list containing the estimate of A, B, sigmac, sigmar
#' @examples
#' \dontrun{
#'   mle.est <- MLE(A.init, B.init, sigmar.init, sigmac.init, X, 200)
#' }
#' @export
MLE <- function(A_init, B_init, sigmar_init, sigmac_init, X, max_iters) {
    .Call(`_StatComp20081_MLE`, A_init, B_init, sigmar_init, sigmac_init, X, max_iters)
}

#' @title The projection estimator
#' @description The projection estimator using Rcpp and RcppArmadillo
#' @param X a tensor of shape (m, n, t), where m is the number of rows, n is the number of columns, and t is the number of observations
#' @return a list containing the estimate of A and B
#' @examples
#' \dontrun{
#'   proj.est <- PROJ(X)
#' }
#' @export
PROJ <- function(X) {
    .Call(`_StatComp20081_PROJ`, X)
}

#' @title The vector autoregressive estimate
#' @description The vector autoregressive estimate using Rcpp and RcppArmadillo
#' @param X a tensor of shape (m, n, t), where m is the number of rows, n is the number of columns, and t is the number of observations
#' @return the estimate of phi = kronecker(B, A)
#' @examples
#' \dontrun{
#'   phi.hat <- VAR(X)
#' }
#' @export
VAR <- function(X) {
    .Call(`_StatComp20081_VAR`, X)
}

#' @title The evaluation metric
#' @description The evaluation metric using Rcpp and RcppArmadillo
#' @param BAHK the Kronecker product of B_hat and A_hat
#' @param A the true value of A
#' @param B the true value of B
#' @return the evaluation metric
#' @examples
#' \dontrun{
#'   B.hat <- matrix(rnorm(9), 3, 3)
#'   A.hat <- matrix(rnorm(4), 2, 2)
#'   B <- matrix(runif(9), 3, 3)
#'   A <- matrix(runif(4), 2, 2)
#'   m <- metric(kronecker(B.hat, A.hat), B, A)
#' }
#' @export
metric <- function(BAHK, B, A) {
    .Call(`_StatComp20081_metric`, BAHK, B, A)
}

