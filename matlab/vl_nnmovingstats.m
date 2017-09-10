function [sm1, sm2, mu, cov, N] = vl_nnmovingstats(x, psm1, psm2, N, varargin)
%VL_NNMOVINGSTATS rolling statistical estimates
%   [MU,COV] = VL_NNMOVINGSTATS(X, PSM1, PSM2, N, VARARGIN) takes in 
%   a tensor X, previous estimates of the shifted first moment PSM1 and 
%   shifted second moment PSM2 and computes updated first and second order
%   shifted statistics SM1 and SM2, together with estimates MU and COV of the
%   mean and covariance of X. X is assumed to be an H x W x C x N tensor,
%   in which we consider each channel to be the feature of interest (so that
%   different spatial locations/batches correspond to different samples). N
%   is an integer tracking the total number of samples used in the estimate
%   so far.
%
%   The function tracks shifted first and second moments (rather than directly
%   tracking the mean and covariance of the features) to avoid issues with 
%   numerical instability, which can occur when the magnitude of the covariance
%   is small relative to the mean.
%
%   VL_NNMOVINGSTATS(..., 'option', value, ...) accepts the following
%   options:
%
%   `K`:: []
%    The "shift" factor used to ensure numerical stability.  If unset, it will
%    be set to the mean of the current sample, X.
%
%   `BESSEL`:: true
%    If true, will apply Bessel correction to the estimate of covariance (to 
%    account for differences in sample/popluation statistics).
%
%   `MAXSAMPLE`:: 1e4
%    Sets a maximum number of samples to be included in the estimation of the
%    statistic update (these samples will be selected uniformly from X).
%
% Copyright (C) 2017 Samuel Albanie and David Novotny
% Licensed under The MIT License [see LICENSE.md for details]

  opts.K = [] ;
  opts.bessel = true ;
  opts.maxSample = 1e4 ;
  [opts, dzdy] = vl_argparsepos(varargin) ;

  % Use the sample mean for K if not provided
  if isempty(opts.K), K = mean(x(:)) ; else, K = opts.K ; end

  numChannels = size(x, 3) ;
  x = permute(x, [3 1 2 4]) ;
  x = reshape(x, numChannels, []) ; % form data matrix
  N = N + size(x, 2) ; % keep track number of samples seen

  sm2_delta = (x - K) * (x - K)' ; % compute update to shifted second moment
  sm2 = psm2 + sm2_delta ;
  sm1 = psm1 + sum(x - K, 2) ;

  if opts.bessel, denom = N - 1 ; else, denom = N ; end
  cov = (sm2 - (sm1 * sm1' / N)) / denom ; % use Bessel correction
  mu = K + (sm1 / N) ;

  assert(isempty(dzdy), 'moving stats only supports forward computation') ;
