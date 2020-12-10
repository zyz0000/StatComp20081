#include <RcppCommon.h>
#include <RcppArmadillo.h>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace Rcpp;
using namespace arma;

//' @title The vector autoregressive estimate
//' @description The vector autoregressive estimate using Rcpp and RcppArmadillo
//' @param X a tensor of shape (m, n, t), where m is the number of rows, n is the number of columns, and t is the number of observations
//' @return the estimate of phi = kronecker(B, A)
//' @examples
//' \dontrun{
//'   phi.hat <- VAR(X)
//' }
//' @export
// [[Rcpp::export]]
RcppExport SEXP VAR(arma::cube X){
  const int n = X.n_rows, m = X.n_cols, t = X.n_slices;
  arma::mat mat1(n*m, n*m, fill::zeros), mat2(n*m, n*m, fill::zeros);

  for (int i=1; i<t; i++){
    arma::mat rslice1 = X.slice(i-1);
    arma::colvec rsr1 = reshape(rslice1, n*m, 1);
    arma::mat rslice2 = X.slice(i);
    arma::colvec rsr2 = reshape(rslice2, n*m, 1);
    mat1 = mat1 + rsr1 * rsr1.t();
    mat2 = mat2 + rsr2 * rsr2.t();
  }

  arma::mat mat1_inv = arma::pinv(mat1);
  arma::mat phi_hat = mat1_inv * mat2;

  return Rcpp::wrap(phi_hat);
}
