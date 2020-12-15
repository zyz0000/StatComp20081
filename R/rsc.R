#' @title rsc
#' @description get the modifiedthe rank of the model,it's the realizationof the rank selection criterion
#' @importFrom MASS ginv
#' @param x the matrix x
#' @param y the matrix y
#' @param min_p_n the minmum of the dimension p and n
#' @param sigma stand error of the model's error term
#' @param n n
#' @param q the rank of the matrix x
#' @return  the norm of the matrix x
#' @examples
#' \dontrun{
#' set.seed(1)
#' A<-matrix(c(runif(80,-2,2)),nrow=16)
#' A[14:16,]=0
#' A[,5]=0
#' x<-matrix(rnorm(16*16),nrow=16)
#' e<-matrix(rnorm(16*5,0,1),nrow=16)
#' sigma=1
#' y=x%*%A+e
#' p=ncol(x)
#' n=ncol(y)
#' m=nrow(x)
#' min_p_n=min(p,n);
#' q=16;
#' k=rsc(x,y,min_p_n,sigma,n,q);
#' }
#' @export
rsc<-function(x,y,min_p_n,sigma,n,q)
{
  
  u=2*(sigma^2)*(sqrt(n)+sqrt(q))
  m=t(x)%*%x
  m_ginv=MASS::ginv(m)
  p=x%*%m_ginv%*%t(x)
  ypy=t(y)%*%p%*%y
  v=eigen(ypy)$vectors
  b_hat=m_ginv%*%t(x)%*%y
  w=b_hat%*%v
  G=t(v)
  value_b_k_penalized_least_squares_estimator=rep(0,min_p_n)
  flag=0
  flag_i=0
  for(i in 1:min_p_n)
  {
    wi=w[,1:i]
    gi=G[1:i,]
    if(i==1)
    {
      bk_hat=wi%*%t(gi)
    }
    if(i>1)
    {
      bk_hat=wi%*%gi
    }
    mm=y-x%*%bk_hat
    trace1=sum(diag(mm%*%t(mm)))
    mmm=sqrt(trace1)
    y_minus_xb_frobenius_norm=mmm
    value_b_k_penalized_least_squares_estimator[i]=y_minus_xb_frobenius_norm^2+u*i
    if(i==1)
    {
      flag=value_b_k_penalized_least_squares_estimator[i]
      flag_i=1
    }
    if(value_b_k_penalized_least_squares_estimator[i] < flag)
    {
      flag=value_b_k_penalized_least_squares_estimator[i]
      flag_i=i
    }
  }
  return(flag_i)
}