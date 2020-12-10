## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ---- cache=TRUE, warning=FALSE-----------------------------------------------
library(grid)
library(ggplot2)
# Show bivariate scatter plot and univariate histogram
p.scatter <- ggplot(iris) + geom_point(aes(x=Sepal.Length, y=Sepal.Width, color=Species))
p.hist.len <- ggplot(iris) + geom_histogram(aes(x=Sepal.Length))
p.hist.wid <- ggplot(iris) + geom_histogram(aes(x=Sepal.Width)) + coord_flip()
grid.newpage()
pushViewport(viewport(layout = grid.layout(3, 3)))
print(p.scatter, vp=viewport(layout.pos.row=2:3, layout.pos.col=1:2))
print(p.hist.len, vp=viewport(layout.pos.row=1, layout.pos.col=1:2))
print(p.hist.wid, vp=viewport(layout.pos.row=2:3, layout.pos.col=3))

## ---- cache=FALSE, warning=FALSE----------------------------------------------
knitr::kable(head(iris, n=5))

## ---- cache=TRUE, warning=FALSE-----------------------------------------------
knitr::kable(summary(iris))

## ---- cache=TRUE, warning=FALSE-----------------------------------------------

library(caret)  #`library(caret)` for cross validation 
library(e1071)  # `library(e1071)` for Naive Bayes
set.seed(922)
trainIndex <- createDataPartition(iris$Species,p=0.8,list=F,times=5)


accuracy <- rep(0, 5)# A numeric vector for restoring validated accuracy
for (fold in 1:5){
  train.idx <- trainIndex[,fold]
  nb <- naiveBayes(Species ~., data=iris)
  pred <- predict(nb, iris[-train.idx, 1:4], type="class")  # prediction on validation dataset
  accuracy[fold] <- confusionMatrix(iris[-train.idx, 5], pred)[["overall"]][1]
}

cat("The mean accuracy of five-fold cross validation is", mean(accuracy))

## ---- cache=TRUE--------------------------------------------------------------
generate.pareto <- function(n, a, b){
  U <- runif(n)
  P <- b * (1 - U)^(-1 / a)
  return (P)
}

set.seed(929)
a <- b <- 2
p <- generate.pareto(1e4, a, b)

hist(p, prob=TRUE, breaks=50, main="Histogram of Pareto(2,2)")
y <- sort(p)
fy <- a * b^a * y^(-(a + 1))
lines(y, fy, col="red", lty=1, lwd=2)

## ---- cache=TRUE--------------------------------------------------------------
generate.fe <- function(n){
  U1 <- runif(n, -1, 1)
  U2 <- runif(n, -1, 1)
  U3 <- runif(n, -1, 1)
  
  U <- ifelse((abs(U3) >= abs(U2) & abs(U3) >= abs(U1)), U2, U3)
  return (U)
}

set.seed(929)
U <- generate.fe(1e4)
hist(U, breaks = 20, prob=TRUE, main=expression(paste("Histogram of ", f[e])))
lines(density(U), lty=1, lwd=2, col="blue")
legend("topleft", "Kernel Density Estimate", lwd=2, col="blue", lty=1, bty="n")

## ---- cache=TRUE--------------------------------------------------------------
generate.repar.pareto <- function(n, beta, r){
  U <- runif(n)
  P <- beta * (1 / ((1 - U)^(1/r)) - 1)
  return (P)
}

set.seed(929)
beta <- 2; r <- 4
P <- generate.repar.pareto(1e3, beta, r)
hist(P, breaks = 50, prob=TRUE, main="Histogram of Pareto(2,4)")
y <- sort(P)
fy <- r * beta^r * (beta + y)^(-(r + 1))
lines(y, fy, lty=1, col="green", lwd=2)

## ---- cache=TRUE--------------------------------------------------------------
set.seed(1013)
m <- 10000
U <- runif(m, 0, pi/3)
theta.hat <- pi/3 * mean(sin(U))
theta.exact <- integrate(function(x){sin(x)}, 0, pi/3)$value
cat("The estimated value of the integral is", theta.hat)
cat("The exact value of the integral is", theta.exact)

## ---- cache=TRUE--------------------------------------------------------------
set.seed(1014)
MC.integral <- function(R = 10000, antithetic = TRUE) {
  u <- runif(R/2, 0, 1)
  if (!antithetic) v <- runif(R/2) else v <- 1 - u
  u <- c(u, v)
  integral <- mean(exp(u))
  integral
}
MC.anti <- MC.mc <- numeric(1000)
for (i in 1:1000){
  MC.anti[i] <- MC.integral(10000, antithetic = TRUE)
  MC.mc[i] <- MC.integral(10000, antithetic = FALSE)
}
cat("The empirical percent of variance reduction using antithetic variate is", 
    (var(MC.mc) - var(MC.anti)) / var(MC.mc))
cat("The theoretical percent of variance reduction using antithetic variate is", 
    (exp(1)^2 - 3*exp(1) + 1) / (-1/2*exp(1)^2 + 2*exp(1) - 3/2))
cat("The estimate using simple Monte Carlo method is", mean(MC.mc), "\n", 
    "the estimate using antithetic variate is", mean(MC.anti), "\n",
    "the theoretical value is", exp(1) - 1, "\n")

## ---- fig.height=4, fig.width=5-----------------------------------------------
g <- function(x){
  return (x^2*exp(-x^2/2) / sqrt(2*pi) * (x>1))
}

curve(g(x), 1, 5, lty=1, col=1, main=expression(g(x)==x^2*e^(-x^2/2) / sqrt(2*pi)))

## ---- fig.height=4, fig.width=5-----------------------------------------------
x <- seq(1, 5, 0.01)
curve(g(x), 1, 5, lty=1, col=1, ylim=c(0,0.8), ylab="")
lines(x, dgamma(x - 1, 2, 2), lty=2, col=2)
lines(x, dnorm(x, 1.5, 1), lty=3, col=3)
legend("right", legend=c("g(x)", "f1(x)", "f2(x)"), lty=c(1,2,3), col=c(1,2,3))

## ---- cache=TRUE, fig.height=4, fig.width=5-----------------------------------
x <- seq(1, 5, 0.01)
plot(g(x) / dgamma(x - 1, 1.5, 2), 
     type="l", lty=1, col=1, lwd=3, xlab="x", ylab="g(x) / f(x)")
lines(g(x) / dnorm(x, 1.5, 1), lty=2, col=2, lwd=3)
legend("right", c("g(x)/f1(x)", "g(x)/f2(x)"), lty=c(1, 2), col=c(1, 2))
axis(1, at=c(1:5))

## ---- cache=TRUE--------------------------------------------------------------
set.seed(1020)
SISE <- function(k){
  M <- 1e4
  m <- M / k
  si <- v <- numeric(k)
  
  g <- function(x) exp(-x) / (1 + x^2)
  f <- function(x, j) exp(-x) / (exp(-(j - 1) / k) - exp(-j / k))
  
  for (j in 1:k){
    r <- runif(m, (j-1) / k, j / k)
    x <- -log(exp(-(j - 1) / k) - (exp(-(j - 1) / k) - exp(-j / k))*r)
    gf <- g(x) / f(x, j)
    si[j] <- mean(gf)
    v[j] <- var(gf)
  }
  return (list(si=si, v=v))
}

result.sise <- SISE(5)
cat("The value of integral is ", sum(result.sise$si), "\n")
cat("The estimated variance is ", mean(result.sise$v))

## ---- cache=TRUE--------------------------------------------------------------
set.seed(1020)
result.sise <- SISE(1)
cat("The value of integral is ", sum(result.sise$si), "\n")
cat("The estimated variance is ", mean(result.sise$v))

## ---- cache=TRUE, fig.height=4, fig.width=5.5---------------------------------
set.seed(1020)
n <- 20
CI <- replicate(1e4, expr={
  x <- rlnorm(n)
  y <- log(x)
  ybar <- mean(y)
  se <- sd(y) / sqrt(n)
  ybar + se * qt(c(0.025, 0.975), df=n-1)
})
L <- CI[1, ]
U <- CI[2, ]
cover <- sum((L <= 0) & (U >= 0))
cat("The empirical covering probability is", cover / ncol(CI))

## ---- cache=TRUE--------------------------------------------------------------
set.seed(1020)
library(dplyr)
m <- 1e4 
n <- 20 
mu <- 2 
cover <- 0 
interval <- replicate(m, expr={
  x <- rchisq(n, df=2)
  mean(x) + qt(c(0.025, 0.975), df=n-1) * sd(x) / sqrt(n)
})
for (i in 1:m){
  cover <- cover + between(mu, interval[, i][1], interval[, i][2])
}
cat("The empirical covering probability is ", cover / m)

## ---- cache=TRUE--------------------------------------------------------------
set.seed(1027)
library(e1071) #for `skewness()` function`
n <- 30
ab <- 1:10
m <- 1e4
cv <- qnorm(1 - 0.05/2, 0, sqrt(6 * (n - 2) / ((n + 1)*(n + 3))))
pwr.beta <- numeric(length(ab))
sktest <- numeric(m)

for (i in 1:length(ab)){
  a <- ab[i]
  sktest <- replicate(m, expr={
    x <- rbeta(n, shape1 = a, shape2 = a)
    as.integer(abs(skewness(x)) >= cv)
  })
  pwr.beta[i] <- mean(sktest)
}

## ---- cache=TRUE--------------------------------------------------------------
pwr.t <- numeric(length(ab))
sktest <- numeric(m)

for (i in 1:length(ab)){
  nu <- ab[i]
  sktest <- replicate(m, expr={
    x <- rt(n, df=nu)
    as.integer(abs(skewness(x)) >= cv)
  })
  pwr.t[i] <- mean(sktest)
}

## ---- cache=TRUE--------------------------------------------------------------
library(ggpubr)
d <- data.frame(df=rep(ab, time=2), 
                pwr=c(pwr.beta, pwr.t),
                dis=rep(c("Beta", "t"), each=10))
ggline(d, x="df", y="pwr", group="dis", linetype="dis", shape="dis", 
       palette = c("#00AFBB", "#E7B800"), 
       xlab="Parameter", ylab="power") + ggthemes::theme_economist()

## ---- cache=TRUE--------------------------------------------------------------
set.seed(1027)
alpha.hat <- 0.055
n <- c(10, 20, 50, 100, 500, 1000)
mu1 <- mu2 <- 0
sigma1 <- 1
sigma2 <- 1.5
m <- 1e4
result <- matrix(0, length(n), 2)

count5test <- function(x, y){
  X <- x - mean(x)
  Y <- y - mean(y)
  outx <- sum(X > max(Y)) + sum(X < min(Y))
  outy <- sum(Y > max(X)) + sum(Y < min(X))
  return (as.integer(max(c(outx, outy)) > 5))
}

for (i in 1:length(n)){
  ni <- n[i]
  tests <- replicate(m, expr={
    x <- rnorm(ni, mu1, sigma1)
    y <- rnorm(ni, mu2, sigma2)
    Fp <- var.test(x, y)$p.value
    Ftest <- as.integer(Fp <= alpha.hat)
    c(count5test(x, y), Ftest)
  })
  result[i, ] <- rowMeans(tests)
}

data.frame(n=n, C5=result[, 1], Fp=result[, 2])

## ----cache=TRUE---------------------------------------------------------------
data(law, package = "bootstrap")
n <- nrow(law)
R.hat <- cor(law$LSAT, law$GPA)
R.jack <- numeric(n)

for (i in 1:n){
  x <- law[-i, ]
  R.jack[i] <- cor(x[, 1], x[, 2])
}

bias.jack <- (n - 1)*(mean(R.jack) - R.hat)
se.jack <- sqrt(var(R.jack) * (n-1)^2 / n)
cat("The jackknife estimate of the bias of correlation statistic is", bias.jack, "\n")
cat("The jackknife estimate of the standard error of correlation statistic is", se.jack)

## ----cache=TRUE---------------------------------------------------------------
set.seed(113)
library(boot)
data(aircondit, package="boot")
x <- aircondit$hours

boot.func <- function(x, i){
  mean(as.matrix(x[i]))
}

b <- boot(
  data = x,
  statistic = boot.func,
  R = 1000
)

boot.ci(b, type=c("norm", "perc", "basic", "bca"))

## ----cache=TRUE---------------------------------------------------------------
qqnorm(x);qqline(x)

## ----cache=TRUE---------------------------------------------------------------
set.seed(1018)
library(boot)
library(bootstrap)
data(scor, package="bootstrap")

n <- nrow(scor)
df <- as.data.frame(scor)
theta.hat <- eigen(cov(df))$value[1] / sum(eigen(cov(df))$value)

## jackknife estimate
theta.j <- rep(0, n)
for (i in 1:n){
  x <- df[-i, ]
  lambda <- eigen(cov(x))$values
  theta.j[i] <- lambda[1] / sum(lambda)
}
bias.jack <- (n - 1) * (mean(theta.j) - theta.hat)
se.jack <- (n - 1) * sqrt(var(theta.j) / n)

cat("The jackknife estimate of bias is", bias.jack, "\n")
cat("The jackknife estimate of standard error is", se.jack)

## ----cache=TRUE, warning=FALSE------------------------------------------------
if (!require(DAAG)) {
  install.packages("DAAG")
  library(DAAG)
}
data(ironslag)
n <- nrow(ironslag)
N <- combn(n, 2) #choose all possible combinations C_{n}^{2}
e1 <- e2 <- e3 <- e4 <- numeric(dim(N)[2])

for (k in 1:dim(N)[2]){
  lto <- N[, k] # leave-two-out index
  y <- ironslag[-lto, 2]
  x <- ironslag[-lto, 1]
  
  J1 <- lm(y ~ x)
  yhat1 <- J1$coef[1] + J1$coef[2]*ironslag[lto, 1]
  e1[k] <- sum((ironslag[lto, 2] - yhat1)^2)
  
  J2 <- lm(y ~ (x + I(x^2)))
  yhat2 <- J2$coef[1] + J2$coef[2] * ironslag[lto, 1] + J2$coef[3] * ironslag[lto, 1]^2
  e2[k] <- sum((ironslag[lto, 2] - yhat2)^2)
  
  J3 <- lm(log(y) ~ x)
  logyhat3 <- J3$coef[1] + J3$coef[2]*ironslag[lto, 1]
  yhat3 <- exp(logyhat3)
  e3[k] <- sum((ironslag[lto, 2] - yhat3)^2)
  
  J4 <- lm(log(y) ~ log(x))
  logyhat4 <- J4$coef[1] + J4$coef[2]*log(ironslag[lto, 1])
  yhat4 <- exp(logyhat4)
  e4[k] <- sum((ironslag[lto, 2] - yhat4)^2)
}

c(mean(e1), mean(e2), mean(e3), mean(e4))

## ----cache=TRUE---------------------------------------------------------------
model <- lm(magnetic~chemical + I(chemical^2), data=ironslag)
summary(model)

## ----cache=TRUE---------------------------------------------------------------
max.extreme.points <- function(x, y){
  X <- x - mean(x)
  Y <- y - mean(y)
  outx <- sum(X > max(Y)) + sum(X < min(Y))
  outy <- sum(Y > max(X)) + sum(Y < min(X))
  max(c(outx, outy))
}


maxout <- function(x, y, R){
  
  # x: a vector representing a sample from population 1
  # y: a vector representing a sample from population 2
  # R: the number of replicates
  # return: a list containing the `maxout` statistic and p value of the permutation test
  
  z <- c(x, y)
  n <- length(x)
  N <- length(z)
  
  stat <- replicate(R, expr={
    k <- sample(c(1:N))
    k1 <- k[1:n]
    k2 <- k[(n+1):N]
    max.extreme.points(z[k1], z[k2])
  })
  stat1 <- max.extreme.points(x, y)
  stat2 <- c(stat, stat1)

  return (list(estimate=stat1, 
               p.val=mean(stat2 >= stat1)))
}


## ----cache=TRUE---------------------------------------------------------------
set.seed(1110)
n <- 30
m <- 60
mu1 <- mu2 <- 0
sigma1 <- sigma2 <- 1
x <- rnorm(n, mu1, sigma1)
y <- rnorm(m, mu2, sigma2)
maxout(x, y, 500)

## ----cache=TRUE---------------------------------------------------------------
sigma1 <- 1
sigma2 <- 2
x <- rnorm(n, mu1, sigma1)
y <- rnorm(m, mu2, sigma2)
maxout(x, y, 500)

## ----cache=TRUE---------------------------------------------------------------
set.seed(1117)

m <- 3 # number of Markov chains
N <- 1000 # the length of each chain
sigma <- 2 # the standard deviation of proposal distribution
X <- matrix(0, nrow=m, ncol=N) # matrix for storing Markov chains
x0 <- 2 # initial value for the chains

rw.Metropolis <- function(sigma, x0, N){
  # sigma is the standard deviation of proposal distribution N(X_t, \sigma^2)
  # x0 is the starting point when implementing sampling
  # N is the length of the chain
  
  x <- numeric(N)
  x[1] <- x0
  u <- runif(N)
  k <- 0
  for (i in 2:N){
    y <- rnorm(1, x[i - 1], sigma)
    if (u[i] <= exp(abs(x[i - 1]) - abs(y))){
      x[i] <- y
    }else{
      x[i] <- x[i - 1]
    }
  }
  
  return (x)
}


for (j in 1:m){
  X[j, ] <- rw.Metropolis(sigma, x0, N)
}

library(coda)
X1 <- as.mcmc(X[1, ])
X2 <- as.mcmc(X[2, ])
X3 <- as.mcmc(X[3, ])
Y <- mcmc.list(X1, X2, X3)
print(gelman.diag(Y))
gelman.plot(Y, col=c(1, 2))

## -----------------------------------------------------------------------------
nAdot <- 444
nBdot <- 132
nOO <- 361
nAB <- 63
n <- nAdot + nBdot + nOO + nAB
rhat <- sqrt(nOO / n)
c1 <- nAdot / n
c2 <- nBdot / n
phat <- (-2*rhat + sqrt((2*rhat)^2 + 4*1*c1)) / 2
qhat <- (-2*rhat + sqrt((2*rhat)^2 + 4*1*c2)) / 2

max.iters <- 20# maximum iterations
p <- phat; q <- qhat # initial value of p and q
result <- matrix(0, max.iters, 3)


for (iter in 1:max.iters){
  
  alpha <- (nAdot * (2*p*(1-p-q))) / (2*p*(1-p-q) + p^2)
  beta <- (nBdot * (2*q*(1-p-q))) / (2*q*(1-p-q) + q^2)
  
  nump <- matrix(c(
    -alpha + 2 * nAdot + nAB,
    2 * nAdot + nAB - alpha,
    -beta + 2 * nBdot + nAB,
    2 * nBdot + nAB + 2*nOO + alpha
  ), nrow = 2, byrow = TRUE)
  
  numq <- matrix(c(
    2 * nAdot + nAB + 2 * nOO + beta,
    -alpha + 2 * nAdot + nAB,
    2 * nBdot + nAB - beta,
    -beta + 2 * nBdot + nAB
  ), nrow = 2, byrow = TRUE)
  
  denom <- matrix(c(
    2 * nAdot + nAB + 2 * nOO + beta,
    2 * nAdot + nAB - alpha,
    2 * nBdot + nAB - beta,
    2 * nBdot + nAB + 2 * nOO + alpha
  ), nrow = 2, byrow = TRUE)
  
  p <- det(nump) / det(denom)
  q <- det(numq) / det(denom)
  r <- 1 - p - q
  loglik <- nAdot*log(p^2 + 2*p*r) + nBdot*log(q^2 + 2*q*r) +
    nOO*log(r^2) + nAB*log(2*p*q)
  
  result[iter, 1] <- p
  result[iter, 2] <- q
  result[iter, 3] <- loglik
}

cat("The MLEs of p and q using EM algorithm are ", p, " and ", q)

## -----------------------------------------------------------------------------
data.frame(p=result[1:10, 1], q=result[1:10, 2], loglik=result[1:10, 3])

## -----------------------------------------------------------------------------
plot(result[, 3][1:10], pch=16, 
     main="Log likelihood vs. iterations", 
     xlab="Number of iterations", ylab="Log likelihood")

## ----eval=FALSE---------------------------------------------------------------
#  formulas <- list(
#    mpg ~ disp,
#    mpg ~ I(1 / disp),
#    mpg ~ disp + wt,
#    mpg ~ I(1 / disp) + wt
#  )

## ----eval=FALSE---------------------------------------------------------------
#  formulas <- list(
#    mpg ~ disp,
#    mpg ~ I(1 / disp),
#    mpg ~ disp + wt,
#    mpg ~ I(1 / disp) + wt
#  )
#  
#  # fit linear models using for loops
#  for (i in 1:4){
#    model <- lm(formulas[[i]], data=mtcars)
#  }
#  
#  # fit linear models using lapply()
#  lapply(formulas, function(formula) lm(formula, data=mtcars))

## ----eval=FALSE---------------------------------------------------------------
#  trials <- replicate(
#    100,
#    t.test(rpois(10, 10), rpois(7, 10)),
#    simplify = FALSE
#  )

## -----------------------------------------------------------------------------
trials <- replicate(
  100,
  t.test(rpois(10, 10), rpois(7, 10)),
  simplify = FALSE
)

# extract the p-value using sapply()
# using anonymous function
p.val.ano <- sapply(trials, function(x) x$p.value)
# using [[
p.val <- sapply(trials, "[[", "p.value")

# check if `p.val.ano` and `p.val` are identical
all(p.val.ano == p.val)

## -----------------------------------------------------------------------------
mylapply <- function(X, FUN, FUN.VALUE, simplify = TRUE){
  val <- Map(function(x) vapply(x, FUN, FUN.VALUE), X)
  if (simplify){
    return (simplify2array(val))
  }
  return (val)
}

## -----------------------------------------------------------------------------
ls <- list(mtcars, iris)
mylapply(ls, mean, numeric(1))

## ----eval=FALSE---------------------------------------------------------------
#  rw.Metropolis.r <- function(sigma, x0, N){
#    # sigma is the standard deviation of proposal distribution N(X_t, \sigma^2)
#    # x0 is the starting point when implementing sampling
#    # N is the length of the chain
#  
#    x <- numeric(N)
#    x[1] <- x0
#    u <- runif(N)
#    k <- 0
#    for (i in 2:N){
#      y <- rnorm(1, x[i - 1], sigma)
#      if (u[i] <= exp(abs(x[i - 1]) - abs(y))){
#        x[i] <- y
#      }else{
#        x[i] <- x[i - 1]
#        k <- k + 1
#      }
#    }
#    accept.rate <- 1 - k / N
#  
#    return (list(x=x, accept.rate=accept.rate))
#  }
#  

## ----eval=FALSE---------------------------------------------------------------
#  library(Rcpp)
#  library(RcppArmadillo)
#  
#  sourceCpp(
#    code = '
#      #include<Rmath.h>
#      #include<RcppCommon.h>
#      #include<RcppArmadillo.h>
#  
#      // [[Rcpp::depends(RcppArmadillo)]]
#      using namespace std;
#      using namespace arma;
#  
#      // [[Rcpp::export]]
#      extern "C" SEXP rw_Metropolis_cpp(
#                      double sigma,
#                      double x0,
#                      int N){
#  
#        vec x(N, fill::zeros);
#        x(0) = x0;
#        vec u = randu<vec>(N);
#        double k = 0.0;
#  
#        for (int i = 1; i < N; i++){
#          double y = ::Rf_rnorm(x(i - 1), sigma);
#          if ( u(i) <= exp(abs(x(i - 1))) / exp(abs(y)) ){
#            x(i) = y;
#          } else{
#            x(i) = x(i - 1);
#            ++k;
#          }
#        }
#        double accept_rate = 1 - (k / N);
#  
#        return Rcpp::List::create(
#          Rcpp::Named("x") = x,
#          Rcpp::Named("accept.rate") = accept_rate
#        );
#      }
#    '
#  )

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(1201)
#  N <- 1e4
#  sigma <- 2
#  x0 <- 5
#  result.r <- rw.Metropolis.r(sigma, x0, N)
#  result.cpp <- rw_Metropolis_cpp(sigma, x0, N)

## ----eval=FALSE---------------------------------------------------------------
#  par(mfrow=c(1,2))
#  plot(result.r$x, type="l", col=2,
#       xlab="iters", ylab="x", ylim=c(-6,6),
#       main="Trace plot: R")
#  plot(result.cpp$x, type="l", col=3,
#       xlab="iters", ylab="x", ylim=c(-6,6),
#       main="Trace plot: C++")

## ----eval=FALSE---------------------------------------------------------------
#  par(mfrow=c(1,2))
#  p <- ppoints(200)
#  Q1 <- quantile(result.r$x, p)
#  qqplot(Q1, result.r$x, pch=16, cex=0.4, main="qqplot: R")
#  abline(0, 1)
#  Q2 <- quantile(result.cpp$x, p)
#  qqplot(Q2, result.cpp$x, pch=16, cex=0.4, main="qqplot: C++")
#  abline(0, 1)
#  par(mfrow=c(1,1))

## ----eval=FALSE---------------------------------------------------------------
#  sigma.list <- c(0.5, 1, 2)
#  acc.rate <- data.frame(sigma=sigma.list,
#                         accR=rep(0, 3),
#                         accCpp=rep(0, 3))
#  for (i in 1:3){
#    acc.rate[i, 2] <- rw.Metropolis.r(sigma.list[i], x0, N)$accept.rate
#    acc.rate[i, 3] <- rw_Metropolis_cpp(sigma.list[i], x0, N)$accept.rate
#  }
#  acc.rate

## ----eval=FALSE---------------------------------------------------------------
#  library(microbenchmark)
#  ts <- microbenchmark(rw.R=rw.Metropolis.r(sigma, x0, N),
#                       rw.Cpp=rw_Metropolis_cpp(sigma, x0, N))
#  summary(ts)

