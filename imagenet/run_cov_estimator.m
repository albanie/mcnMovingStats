function run_cov_estimator
% RUN_COV_ESTIMATOR estimate covariance on imagenet training data
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  gpus = 1 ;
  batchSize = 32 ;
  useCached = 1 ; % load results from cache if available

  importedModels = {
  {'SE-ResNet-50-mcn', {'conv2_3_relu','conv3_4_relu', 'conv4_6_relu', 'conv5_3_relu'}}, ...
  {'SE-ResNet-101-mcn', {'conv2_3_relu','conv3_4_relu', 'conv4_23_relu', 'conv5_3_relu'}}...
  {'SE-ResNet-152-mcn', {'conv2_3_relu','conv3_8_relu', 'conv4_36_relu', 'conv5_3_relu'}} ...
  } ;

  for ii = 1:numel(importedModels)
    modelPair = importedModels{ii} ;
    imagenet_eval(modelPair, batchSize, gpus, useCached) ;
  end

% -------------------------------------------------------
function imagenet_eval(model, batchSize, gpus, useCached)
% -------------------------------------------------------
  imagenet_cov_estimation('model', model{1}, 'batchSize', batchSize, ...
                          'gpus', gpus, 'useCached', useCached, ...
                          'targetFeats', model{2}) ;
