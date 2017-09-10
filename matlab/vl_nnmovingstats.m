function [mu, cov] = vl_nnmovingstats(x, pMu, pCov, N, varargin)
%VL_NNMOVINGSTATS rolling mean and covariance estimates
% [MU,COV] = VL_NNMOVINGSTATS(X, PMU, PCOV) 

  opts.K = [] ;
  opts.maxSample = 1e4 ;
  opts.bessel = true ;
  [opts, dzdy] = vl_argparsepos(varargin) ;

  % Use the sample mean for K if not provided
  K = mean(x_(:)) ;

  numChannels = size(x, 3) ;
  x = permute(x, [3 1 2 4]) ;
  x = reshape(x, numChannels, []) ;
  N = N + size(x, 2) ; % update number of samples seen

  batchCov = (x - K) * (x - K)' ;
  M12 = M12 + batchCov ;
  residuals = residuals + sum(x - K, 2) ;

  if opts.bessel, denom = N - 1 ; else, denom = N ; end
  cov = (M12 - (residuals * residuals' / N)) / denom ; % use Bessel correction
  mu = K + (residuals / N) ;

  assert(isempty(dzdy), 'moving stats only supports forward computation') ;
