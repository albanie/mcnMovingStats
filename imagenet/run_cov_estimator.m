function run_cov_estimator
% RUN_COV_ESTIMATOR estimate covariance on imagenet training data
%
% Copyright (C) 2017 Samuel Albanie and David Novotny
% Licensed under The MIT License [see LICENSE.md for details]

  gpus = 1 ;
  batchSize = 32 ;
  useCached = 1 ; % load results from cache if available

  % select features to track 
  targetFeats = {'res2c_relu', 'res3d_relu', 'res4f_relu'} ;

  % select models to compute statistics of these features for 
  models = {
    {'imagenet-resnet-50-dag', targetFeats}, ...
  } ;

  for ii = 1:numel(models)
    modelPair = models{ii} ;
    imagenet_eval(modelPair, batchSize, gpus, useCached) ;
  end

% -------------------------------------------------------
function imagenet_eval(model, batchSize, gpus, useCached)
% -------------------------------------------------------
  imagenet_cov_estimation('model', model{1}, 'batchSize', batchSize, ...
                          'gpus', gpus, 'useCached', useCached, ...
                          'targetFeats', model{2}) ;
