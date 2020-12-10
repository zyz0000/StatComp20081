#include <RcppCommon.h>
#include <RcppArmadillo.h>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace Rcpp;
using namespace arma;

//' @title The projection estimator
//' @description The projection estimator using Rcpp and RcppArmadillo
//' @param X a tensor of shape (m, n, t), where m is the number of rows, n is the number of columns, and t is the number of observations
//' @return a list containing the estimate of A and B
//' @examples
//' \dontrun{
//'   proj.est <- PROJ(X)
//' }
//' @export
// [[Rcpp::export]]
extern "C" SEXP PROJ(arma::cube X){
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

    int cnt = 0;
    arma::mat phi_trans(n*m, n*m, fill::zeros);

    do{
      for (int col=0; col<n; col++){
        int first_col = col*m;
        int last_col = (col + 1)*m - 1;
        for (int row=0; row<m; row++){
          int first_row = row*n;
          int last_row = (row + 1)*n - 1;
          arma::mat p = phi_hat(span(first_row, last_row),
                                span(first_col, last_col));
          phi_trans.col(cnt) = reshape(p, n*m, 1);
          cnt += 1;
        }
      }
    }while(cnt <= n*m - 1);

    arma::mat U0;
    arma::vec s0;
    arma::mat V0;
    svd(U0, s0, V0, phi_trans);

    arma::mat U = U0;
    arma::vec s = s0;
    arma::mat V = V0;

    double s_val = s0(0);
    arma::mat B_opt = s_val * U.col(0);
    arma::mat A_opt = s_val * V.col(0);

    arma::mat B_est = reshape(B_opt, n, m);
    arma::mat A_est = reshape(A_opt, m, n);

    return Rcpp::List::create(
      Rcpp::Named("B.est") = B_est,
      Rcpp::Named("A.est") = A_est
    );
}
