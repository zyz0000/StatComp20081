#include <RcppCommon.h>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace Rcpp;
using namespace arma;

//' @title The iterative least square estimator
//' @description The iterative least square estimator using Rcpp and RcppArmadillo
//' @param X a tensor of shape (m, n, t), where m is the number of rows, n is the number of columns, and t is the number of observations
//' @param A_init the initial value of A
//' @param B_init the initial value of B
//' @param max_iters the maximum iterations
//' @return a list containing the estimate of A and B
//' @examples
//' \dontrun{
//'   ils.est <- ILS(X, A_init, B_init, 200)
//' }
//' @export
// [[Rcpp::export]]
extern "C" SEXP ILS(arma::cube X,
           arma::mat A_init,
           arma::mat B_init,
           int max_iters){

    const int m = X.n_rows, n = X.n_cols, t = X.n_slices;
    arma::mat A = A_init, B = B_init;

    for (int i = 0; i < max_iters; i++){
      arma::mat mat1(n, n, fill::zeros), mat2(n, n, fill::zeros);
      arma::mat mat3(m, m, fill::zeros), mat4(m, m, fill::zeros);

      for (int j = 1; j < t; j++){
        mat1 += trans(X.slice(j)) * A * X.slice(j-1);
        mat2 += trans(X.slice(j)) * A.t() * A * X.slice(j-1);
      }
      B = mat1 * arma::pinv(mat2);

      for (int k = 1; k < t; k++){
        mat3 += X.slice(k) * B * trans(X.slice(k-1));
        mat4 += X.slice(k-1) * B.t() * B * trans(X.slice(k-1));
      }
      A = mat3 * arma::pinv(mat4);
      A = A / (arma::norm(A, "fro"));
    }

    return Rcpp::List::create(
      Rcpp::Named("B.est") = B,
      Rcpp::Named("A.est") = A
    );
}
