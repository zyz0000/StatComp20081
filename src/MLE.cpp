#include <RcppCommon.h>
#include <RcppArmadillo.h>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace Rcpp;
using namespace arma;

//' @title The maximum likelihood estimator
//' @description The maximum likelihood estimator using Rcpp and RcppArmadillo
//' @param A_init the initial value of A
//' @param B_init the initial value of B
//' @param sigmar_init the initial value of sigmar
//' @param sigmac_init the initial value of sigmac
//' @param X a tensor of shape (m, n, t), where m is the number of rows, n is the number of columns, and t is the number of observations
//' @param max_iters the maximum iterations
//' @return a list containing the estimate of A, B, sigmac, sigmar
//' @examples
//' \dontrun{
//'   mle.est <- MLE(A.init, B.init, sigmar.init, sigmac.init, X, 200)
//' }
//' @export
// [[Rcpp::export]]
extern "C" SEXP MLE(arma::mat A_init,
           arma::mat B_init,
           arma::mat sigmar_init,
           arma::mat sigmac_init,
           arma::cube X,
           int max_iters){

    const int n = B_init.n_rows, m = A_init.n_rows, t = X.n_slices;
    arma::mat A = A_init, B = B_init, sigmac = sigmac_init, sigmar = sigmar_init;

    for (int i = 0; i < max_iters; i++){
      arma::mat mat1(m, m, fill::zeros), mat2(m, m, fill::zeros), mat6(m, m, fill::zeros);
      arma::mat mat3(n, n, fill::zeros), mat4(n, n, fill::zeros), mat5(n, n, fill::zeros);

      for (int j = 1; j < t; j++){
        mat1 += X.slice(j) * arma::pinv(sigmac) * B * trans(X.slice(j - 1));
        mat2 += X.slice(j - 1) * B.t() * arma::pinv(sigmac) * B * trans(X.slice(j-1));
      }
      A = mat1 * arma::pinv(mat2);

      for (int k = 1; k < t; k++){
        mat3 += trans(X.slice(k)) * arma::pinv(sigmar) * A * X.slice(k-1);
        mat4 += trans(X.slice(k-1)) * A.t() * arma::pinv(sigmar) * A * X.slice(k-1);
      }
      B = mat3 * arma::pinv(mat4);

      for (int l = 1; l < t; l++){
        arma::mat R = X.slice(l) - A * X.slice(l-1) * B.t();
        mat5 += R.t() * arma::pinv(sigmar) * R;
        mat6 += R * arma::pinv(sigmac) * R.t();
      }
      sigmac = mat5 / (m * (t - 1));
      sigmar = mat6 / (n * (t - 1));

      A = A / (arma::norm(A, "fro"));
      sigmar = sigmar / (arma::norm(sigmar, "fro"));
    }


    return Rcpp::List::create(
      Rcpp::Named("A.est") = A,
      Rcpp::Named("B.est") = B,
      Rcpp::Named("sigmar.est") = sigmar,
      Rcpp::Named("sigmac.est") = sigmac
    );
}
