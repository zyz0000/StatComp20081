#include <math.h>
#include <RcppCommon.h>
#include <RcppArmadillo.h>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace Rcpp;
using namespace arma;

//' @title The evaluation metric
//' @description The evaluation metric using Rcpp and RcppArmadillo
//' @param BAHK the Kronecker product of B_hat and A_hat
//' @param A the true value of A
//' @param B the true value of B
//' @return the evaluation metric
//' @examples
//' \dontrun{
//'   B.hat <- matrix(rnorm(9), 3, 3)
//'   A.hat <- matrix(rnorm(4), 2, 2)
//'   B <- matrix(runif(9), 3, 3)
//'   A <- matrix(runif(4), 2, 2)
//'   m <- metric(kronecker(B.hat, A.hat), B, A)
//' }
//' @export
// [[Rcpp::export]]
extern "C" SEXP metric(arma::mat BAHK,
            arma::mat B,
            arma::mat A){

  arma::mat BAK = arma::kron(B, A);
  arma::mat diff = BAHK - BAK;
  double norm = arma::norm(diff, "fro");
  return (Rcpp::wrap(log(pow(norm, 2))));

}
