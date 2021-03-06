% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{metric}
\alias{metric}
\title{The evaluation metric}
\usage{
metric(BAHK, B, A)
}
\arguments{
\item{BAHK}{the Kronecker product of B_hat and A_hat}

\item{B}{the true value of B}

\item{A}{the true value of A}
}
\value{
the evaluation metric
}
\description{
The evaluation metric using Rcpp and RcppArmadillo
}
\examples{
\dontrun{
  B.hat <- matrix(rnorm(9), 3, 3)
  A.hat <- matrix(rnorm(4), 2, 2)
  B <- matrix(runif(9), 3, 3)
  A <- matrix(runif(4), 2, 2)
  m <- metric(kronecker(B.hat, A.hat), B, A)
}
}
