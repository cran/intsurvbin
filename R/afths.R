#' Horseshoe shrinkage prior in Bayesian survival regression

#'

#'

#' This function employs the algorithm provided by van der Pas et. al. (2016) for

#' log normal Accelerated Failure Rate (AFT) model to fit survival regression. The censored observations are updated

#' according to the data augmentation of approach of Tanner and Wong (1984).

#'

#'  The model is:

#'  \eqn{t_i} is response,

#'  \eqn{c_i} is censored time,

#'  \eqn{t_i^* = \min_(t_i, c_i)} is observed time,

#'  \eqn{w_i} is censored data, so \eqn{w_i = \log t_i^*} if \eqn{t_i} is event time and

#'  \eqn{w_i = \log t_i^*} if \eqn{t_i} is right censored

#'  \eqn{\log t_i=X\beta+\epsilon, \epsilon \sim N(0,\sigma^2)}.

#'

#' @references Maity, A. K., Carroll, R. J., and Mallick, B. K. (2019) 
#'             "Integration of Survival and Binary Data for Variable Selection and Prediction: 
#'             A Bayesian Approach", 
#'             Journal of the Royal Statistical Society: Series C (Applied Statistics).
#'

#'@param ct survival response, a \eqn{n*2} matrix with first column as response and second column as right censored indicator,

#'1 is event time and 0 is right censored.

#'@param X Matrix of covariates, dimension \eqn{n*p}.

#'@param method.tau Method for handling \eqn{\tau}. Select "truncatedCauchy" for full

#' Bayes with the Cauchy prior truncated to [1/p, 1], "halfCauchy" for full Bayes with

#' the half-Cauchy prior, or "fixed" to use a fixed value (an empirical Bayes estimate,

#' for example).

#'@param tau  Use this argument to pass the (estimated) value of \eqn{\tau} in case "fixed"

#' is selected for method.tau. Not necessary when method.tau is equal to"halfCauchy" or

#' "truncatedCauchy". The default (tau = 1) is not suitable for most purposes and should be replaced.

#'@param method.sigma Select "Jeffreys" for full Bayes with Jeffrey's prior on the error

#'variance \eqn{\sigma^2}, or "fixed" to use a fixed value (an empirical Bayes

#'estimate, for example).

#'@param Sigma2 A fixed value for the error variance \eqn{\sigma^2}. Not necessary

#'when method.sigma is equal to "Jeffreys". Use this argument to pass the (estimated)

#'value of Sigma2 in case "fixed" is selected for method.sigma. The default (Sigma2 = 1)

#'is not suitable for most purposes and should be replaced.

#'@param burn Number of burn-in MCMC samples. Default is 1000.

#'@param nmc Number of posterior draws to be saved. Default is 5000.

#'@param thin Thinning parameter of the chain. Default is 1 (no thinning).

#'@param alpha Level for the credible intervals. For example, alpha = 0.05 results in

#'95\% credible intervals.

#'@param Xtest test design matrix.

#'@param cttest test survival response.



#'

#'@return 
#' \item{SurvivalHat}{Predictive survival probability}
#' \item{LogTimeHat}{Predictive log time}
#' \item{Beta.sHat}{Posterior mean of Beta, a \eqn{p} by 1 vector}
#' \item{LeftCI.s}{The left bounds of the credible intervals}
#' \item{RightCI.s}{The right bounds of the credible intervals}
#' \item{Beta.sMedian}{Posterior median of Beta, a \eqn{p} by 1 vector}
#' \item{LambdaHat}{Posterior samples of \eqn{\lambda}, a \eqn{p*1} vector}
#' \item{Sigma2Hat}{Posterior mean of error variance \eqn{\sigma^2}. If method.sigma =
#' "fixed" is used, this value will be equal to the user-selected value of Sigma2
#' passed to the function}
#' \item{TauHat}{Posterior mean of global scale parameter tau, a positive scalar}
#' \item{Beta.sSamples}{Posterior samples of \eqn{\beta}}
#' \item{TauSamples}{Posterior samples of \eqn{\tau}}
#' \item{Sigma2Samples}{Posterior samples of Sigma2}
#'

#' @importFrom stats dnorm pnorm rbinom rnorm var

#' @examples

#' burnin <- 500
#' nmc    <- 500
#' thin <- 1
#' y.sd   <- 1  # standard deviation of the response
#' 
#' p <- 100  # number of predictors
#' ntrain <- 100  # training size
#' ntest  <- 50   # test size
#' n <- ntest + ntrain  # sample size
#' q <- 10   # number of true predictos
#' 
#' beta.t <- c(sample(x = c(1, -1), size = q, replace = TRUE), rep(0, p - q))  # randomly assign sign
#' 
#' Sigma <- matrix(0.9, nrow = p, ncol = p)
#' for(j in 1:p)
#' {
#' Sigma[j, j] <- 1
#' }
#' 
#' x <- mvtnorm::rmvnorm(n, mean = rep(0, p), sigma = Sigma)    # correlated design matrix
#' 
#' tmean <- x %*% beta.t
#' yCorr <- 0.5
#' yCov <- matrix(c(1, yCorr, yCorr, 1), nrow = 2)
#' 
#' 
#' y <- mvtnorm::rmvnorm(n, sigma = yCov)
#' t <- y[, 1] + tmean
#' X <- scale(as.matrix(x))  # standarization
#' 
#' t <- as.numeric(as.matrix(c(t)))
#' T <- exp(t)   # AFT model
#' C <- rgamma(n, shape = 1.75, scale = 3)   # 42% censoring time
#' time <- pmin(T, C)  # observed time is min of censored and true
#' status = time == T   # set to 1 if event is observed
#' ct <- as.matrix(cbind(time = time, status = status))  # censored time
#' 
#' 
#' # Training set
#' cttrain <- ct[1:ntrain, ]
#' Xtrain  <- X[1:ntrain, ]
#' 
#' # Test set
#' cttest <- ct[(ntrain + 1):n, ]
#' Xtest  <- X[(ntrain + 1):n, ]
#' 
#' posterior.fit.aft <- afths(ct = cttrain, X = Xtrain, method.tau = "halfCauchy",
#'                              method.sigma = "Jeffreys", burn = burnin, nmc = nmc, thin = 1,
#'                              Xtest = Xtest, cttest = cttest)
#'                              
#' posterior.fit.aft$Beta.sHat

#' @export





# 15 June 2016

# Use Polya Gamma data augmentation approach instead of Albert and Chib (1993) 





afths <- function(ct, X, method.tau = c("fixed", "truncatedCauchy","halfCauchy"), tau = 1,
                    method.sigma = c("fixed", "Jeffreys"), Sigma2 = 1,
                    burn = 1000, nmc = 5000, thin = 1, alpha = 0.05,
                    Xtest = NULL, cttest = NULL)
{
  
  method.tau = match.arg(method.tau)
  
  method.sigma = match.arg(method.sigma)
  
  ptm=proc.time()
  niter <- burn+nmc
  effsamp=(niter -burn)/thin
  
  
  # X <- as.matrix(bdiag(X, X))
  n=nrow(X)
  p <- ncol(X)
  if(is.null(Xtest))
  {
    Xtest <- X
    ntest <- n
    # cttest<- ct
  } else {
    ntest <- nrow(Xtest)
  }
  
  
  
  time         <- ct[, 1]
  status       <- ct[, 2]
  censored.id  <- which(status == 0)
  n.censored   <- length(censored.id)  # number of censored observations
  X.censored   <- X[censored.id, ]
  X.observed   <- X[-censored.id, ]
  y.s <- logtime <- log(time)   # for coding convenience, since the whole code is written with y
  y.s.censored <- y.s[censored.id]
  y.s.observed <- y.s[-censored.id]
  
  
  timetest         <- cttest[, 1]
  statustest       <- cttest[, 2]
  censored.idtest  <- which(statustest == 0)
  n.censoredtest   <- length(censored.idtest)  # number of censored observations
  y.stest <- logtimetest <- log(timetest)   # for coding convenience, since the whole code is written with y
  
  
  ## parameters ##
  beta.s   <- rep(0, p);
  lambda   <- rep(1, p);
  sigma_sq <- Sigma2;
  adapt.par  <- c(100, 20, 0.5, 0.75)
  prop.sigma <- 1
  
  
  ## output ##
  beta.sout          <- matrix(0, p, effsamp)
  lambdaout          <- matrix(0, p, effsamp)
  tauout             <- rep(0, effsamp)
  sigmaSqout         <- rep(1, effsamp)
  predsurvout        <- matrix(0, ntest, effsamp)
  logtimeout         <- matrix(0, ntest, effsamp)
  loglikelihood.sout <- rep(0, effsamp)
  likelihood.sout    <- matrix(0, n, effsamp)
  y.sout             <- matrix(0, n, effsamp)
  trace              <- array(dim = c(niter, length(sigma_sq)))
  
  
  
  ## start Gibbs sampling ##
  
  message("Markov chain monte carlo is running")
  
  for(i in 1:niter)
  {
    
    ################################
    ####### survival part ##########
    ################################
    
    mean.impute <- X.censored %*% beta.s
    sd.impute   <- sqrt(sigma_sq)
    ## update censored data ##
    time.censored <- msm::rtnorm(n.censored, mean = mean.impute, sd = sd.impute, lower = y.s.censored)
    # truncated at log(time) for censored data
    y.s[censored.id] <- time.censored
    
    
    
    ## which algo to use ##
    if(p > n)
    {
      bs  = bayesreg.sample_beta(X, z = y.s, mvnrue = FALSE, b0 = 0, sigma2 = sigma_sq, tau2 = tau^2, 
                                 lambda2 = lambda^2, omega2  = matrix(1, n, 1), XtX = NA)
      beta.s <- bs$x
      
    } else {
      
      bs  = bayesreg.sample_beta(X, z = y.s, mvnrue = TRUE, b0 = 0, sigma2 = sigma_sq, tau2 = tau^2, 
                                 lambda2 = lambda^2, omega2  = matrix(1, n, 1), XtX = NA)
      beta.s <- bs$x
    }
    
    
    
    # ## update sigma_sq ##
    # if(method.sigma == "Jeffreys"){
    #   
    #   E_1=max(t(y.s-X%*%beta.s)%*%(y.s-X%*%beta.s),(1e-10))
    #   E_2=max(sum(beta.s^2/(tau*lambda)^2),(1e-10))
    #   
    #   sigma_sq= 1/stats::rgamma(1, (n + p)/2, scale = 2/(E_1+E_2))
    # }
    
    ## Update Sigma using Metropolis Hastings scheme
    sigma2.current <- sigma_sq
    theta.current  <- log(sigma2.current)
    trace[i, ]     <- theta.current
    
    # adjust sigma for proposal distribution (taken from Metro_Hastings() function)
    if (i > adapt.par[1] && i%%adapt.par[2] == 0 && i < (adapt.par[4] * niter))
    {
      len <- floor(i * adapt.par[3]):i
      t   <- trace[len, ]
      nl  <- length(len)
      p.sigma <- (nl - 1) * var(t)/nl
      p.sigma <- MHadaptive::makePositiveDefinite(p.sigma)
      if (!(0 %in% p.sigma))
        prop.sigma <- p.sigma
    }
    
    theta.proposed  <- rnorm(1, mean = theta.current, sd = sqrt(prop.sigma))
    sigma2.proposed <- exp(theta.proposed)
    y.s.tilde       <- y.s - X %*% beta.s
    kernel          <- min(1, exp(sum(dnorm(y.s.tilde, sd = sqrt(sigma2.proposed), log = TRUE)) -
                                    sum(dnorm(y.s.tilde, sd = sqrt(sigma2.current), log = TRUE)) +
                                    dnorm(theta.proposed, log = TRUE) - 
                                    dnorm(theta.current,  log = TRUE) + 
                                    abs(theta.proposed) - abs(theta.current)))
    # dnorm(theta.current, log = TRUE) - 
    # dnorm(theta.proposed, log = TRUE))) 
    if(rbinom(1, size = 1, kernel) == 1)
    {
      theta.current <- theta.proposed
      sigma2.current <- sigma2.proposed
    }
    sigma_sq <- sigma2.current 
    
    
    
    
    beta <- c(beta.s)  # all \beta's together
    Beta <- matrix(beta, ncol = p, byrow = TRUE)
    Beta[1, ] <- Beta[1, ]/sqrt(sigma_sq)
    
    ## update lambda_j's in a block using slice sampling ##
    eta = 1/(lambda^2)
    upsi = stats::runif(p,0,1/(1+eta))
    tempps = apply(Beta^2, 2, sum)/(2*tau^2)
    ub = (1-upsi)/upsi
    # now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
    Fub = stats::pgamma(ub, (1 + 1)/2, scale = 1/tempps)
    Fub[Fub < (1e-4)] = 1e-4;  # for numerical stability
    up = stats::runif(p,0,Fub)
    eta <- stats::qgamma(up, (1 + 1)/2, scale=1/tempps)
    lambda = 1/sqrt(eta);
    
    ## update tau ##
    ## Only if prior on tau is used
    if(method.tau == "halfCauchy"){
      tempt <- sum(apply(Beta^2, 2, sum)/(2*lambda^2))
      et = 1/tau^2
      utau = stats::runif(1,0,1/(1+et))
      ubt = (1-utau)/utau
      Fubt = stats::pgamma(ubt,(p+1)/2,scale=1/tempt)
      Fubt = max(Fubt,1e-8) # for numerical stability
      ut = stats::runif(1,0,Fubt)
      et = stats::qgamma(ut,(p+1)/2,scale=1/tempt)
      tau = 1/sqrt(et)
    }#end if
    
    if(method.tau == "truncatedCauchy"){
      tempt <- sum(apply(Beta^2, 2, sum)/(2*lambda^2))
      et = 1/tau^2
      utau = stats::runif(1,0,1/(1+et))
      ubt_1=1
      ubt_2 = min((1-utau)/utau,p^2)
      Fubt_1 = stats::pgamma(ubt_1,(p+1)/2,scale=1/tempt)
      Fubt_2 = stats::pgamma(ubt_2,(p+1)/2,scale=1/tempt)
      #Fubt = max(Fubt,1e-8) # for numerical stability
      ut = stats::runif(1,Fubt_1,Fubt_2)
      et = stats::qgamma(ut,(p+1)/2,scale=1/tempt)
      tau = 1/sqrt(et)
    }
    
    
    mean <- Xtest %*% beta.s
    sd   <- sqrt(sigma_sq)
    predictive.survivor <- pnorm(mean/sd, lower.tail = FALSE)
    logt <- Xtest %*% beta.s
    
    
    
    ## Prediction ##
    mean.stest          <- Xtest %*% beta.s
    sd.stest            <- sqrt(sigma_sq)
    predictive.survivor <- pnorm((y.stest - mean.stest)/sd.stest, lower.tail = FALSE)
    logt                <- mean.stest
    
    ## Following is required for DIC computation
    loglikelihood.s <- sum(c(dnorm(y.s.observed, mean = X.observed %*% beta.s, sd = sqrt(sigma_sq),
                                   log = TRUE),
                             log(1 - pnorm(y.s.censored, mean = X.censored %*% beta.s,
                                           sd = sqrt(sigma_sq)))))
    # loglikelihood.s <- sum(c(dnorm(y.s[-censored.id], mean = X.observed %*% beta.s, sd = sqrt(sigma_sq), 
    #                                log = TRUE),
    #                          dtnorm(y.s[censored.id], mean = X.censored %*% beta.s, sd = sqrt(sigma_sq),
    #                                 lower = y.s.censored, log = TRUE)))
    
    
    ## Following is required for DIC computation
    likelihood.s <- c(dnorm(y.s.observed, mean = X.observed %*% beta.s, sd = sqrt(sigma_sq)),
                      pnorm(y.s.censored, mean = X.censored %*% beta.s, sd = sqrt(sigma_sq),
                            lower.tail = FALSE))
    
    
    if (i%%500 == 0)
    {
      message("iteration = ", i)
    }
    
    
    
    if(i > burn && i%%thin== 0)
    {
      beta.sout[ ,(i-burn)/thin]     <- beta.s
      lambdaout[ ,(i-burn)/thin]     <- lambda
      tauout[(i - burn)/thin]        <- tau
      sigmaSqout[(i - burn)/thin]    <- sigma_sq
      predsurvout[ ,(i - burn)/thin] <- predictive.survivor
      logtimeout[, (i - burn)/thin]  <- logt
      loglikelihood.sout[(i - burn)/thin] <- loglikelihood.s
      likelihood.sout[, (i - burn)/thin]  <- likelihood.s
      y.sout[, (i - burn)/thin] <- y.s
    }
  }
  
  
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  pMean.s   <- apply(beta.sout,1, mean)
  pMedian.s <- apply(beta.sout,1,stats::median)
  pLambda   <- apply(lambdaout, 1, mean)
  pSigma= mean(sigmaSqout)
  pTau=mean(tauout)
  pPS <- apply(predsurvout, 1, mean)
  pLogtime  <- apply(logtimeout, 1, mean)
  pLoglikelihood.s <- mean(loglikelihood.sout)
  pLikelihood.s    <- apply(likelihood.sout, 1, mean)
  py.s             <- apply(y.sout, 1, mean)
  
  
  loglikelihood.posterior.s <- sum(c(dnorm(y.s.observed, mean = X.observed %*% pMean.s, sd = sqrt(pSigma), log = TRUE),
                                     log(1 - pnorm(y.s.censored, mean = X.censored %*% pMean.s,
                                                   sd = sqrt(pSigma)))))
  # loglikelihood.posterior.s <- sum(c(dnorm(py.s[-censored.id], mean = X.observed %*% pMean.s, sd = sqrt(pSigma), 
  #                                log = TRUE),
  #                          dtnorm(py.s[censored.id], mean = X.censored %*% pMean.s, sd = sqrt(pSigma),
  #                                 lower = y.s.censored, log = TRUE)))
  DIC.s <- -4 * pLoglikelihood.s + 2 * loglikelihood.posterior.s
  WAIC <- -2 * (sum(log(pLikelihood.s)) - 
                  2 * (sum(log(pLikelihood.s)) - pLoglikelihood.s))
  
  
  #construct credible sets
  left <- floor(alpha*effsamp/2)
  right <- ceiling((1-alpha/2)*effsamp)
  
  beta.sSort <- apply(beta.sout, 1, sort, decreasing = F)
  left.spoints <- beta.sSort[left, ]
  right.spoints <- beta.sSort[right, ]
  
  
  result=list("SurvivalHat" = pPS, "LogTimeHat" = pLogtime, "Beta.sHat"= pMean.s, 
              "LeftCI.s" = left.spoints, "RightCI.s" = right.spoints,
              "Beta.sMedian" = pMedian.s, 
              "LambdaHat" = pLambda,
              "Sigma2Hat"=pSigma,"TauHat"=pTau, "Beta.sSamples" = beta.sout, 
              "TauSamples" = tauout, "Sigma2Samples" = sigmaSqout, "DIC.s" = DIC.s)
  return(result)
}




# ============================================================================================================================
# Sample the regression coefficients
bayesreg.sample_beta <- function(X, z, mvnrue, b0, sigma2, tau2, lambda2, omega2, XtX)
{
  alpha  = (z - b0)
  Lambda = sigma2 * tau2 * lambda2
  sigma  = sqrt(sigma2)
  
  # Use Rue's algorithm
  if (mvnrue)
  {
    # If XtX is not precomputed
    if (any(is.na(XtX)))
    {
      omega = sqrt(omega2)
      X0    = apply(X,2,function(x)(x/omega))
      bs    = bayesreg.fastmvg2_rue(X0/sigma, alpha/sigma/omega, Lambda)
    }
    
    # XtX is precomputed (Gaussian only)
    else {
      bs    = bayesreg.fastmvg2_rue(X/sigma, alpha/sigma, Lambda, XtX/sigma2)
    }
  }
  
  # Else use Bhat. algorithm
  else
  {
    omega = sqrt(omega2)
    X0    = apply(X,2,function(x)(x/omega))
    bs    = bayesreg.fastmvg_bhat(X0/sigma, alpha/sigma/omega, Lambda)
  }
  
  return(bs)
}


# ============================================================================================================================
# function to generate multivariate normal random variates using Rue's algorithm
bayesreg.fastmvg2_rue <- function(Phi, alpha, d, PtP = NA)
{
  Phi   = as.matrix(Phi)
  alpha = as.matrix(alpha)
  r     = list()
  
  # If PtP not precomputed
  if (any(is.na(PtP)))
  {
    PtP = t(Phi) %*% Phi
  }
  
  p     = ncol(Phi)
  if (length(d) > 1)
  {
    Dinv  = diag(as.vector(1/d))
  }
  else
  {
    Dinv   = 1/d
  }
  L     = t(chol(PtP + Dinv))
  v     = forwardsolve(L, t(Phi) %*% alpha)
  r$m   = backsolve(t(L), v)
  w     = backsolve(t(L), rnorm(p,0,1))
  
  r$x   = r$m + w
  return(r)
}


# ============================================================================================================================
# function to generate multivariate normal random variates using Bhat. algorithm
bayesreg.fastmvg_bhat <- function(Phi, alpha, d)
{
  d     = as.matrix(d)
  p     = ncol(Phi)
  n     = nrow(Phi)
  r     = list()
  
  u     = as.matrix(rnorm(p,0,1)) * sqrt(d)
  delta = as.matrix(rnorm(n,0,1))
  v     = Phi %*% u + delta
  Dpt   = (apply(Phi, 1, function(x)(x*d)))
  W     = Phi %*% Dpt + diag(1,n)
  w     = solve(W,(alpha-v))
  
  r$x   = u + Dpt %*% w
  r$m   = r$x
  
  return(r)
}
