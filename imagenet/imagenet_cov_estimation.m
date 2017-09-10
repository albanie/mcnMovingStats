function imagenet_cov_estimation(varargin)

  opts.gpus = 3 ;
  opts.useCached = 1 ;
  opts.batchSize = 128 ;
  opts.sampleSize =  5000 ;
  opts.model = 'SE-ResNet-50-mcn' ;
  opts.targetFeats = {'conv5_3_relu'} ;
  opts.featDir = fullfile(vl_rootnn, 'data/featCovs') ;
  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts.dataDir = fullfile(vl_rootnn, 'data/datasets/ILSVRC2012') ;
  opts.imdbPath = fullfile(vl_rootnn, 'data', 'imagenet12', 'imdb.mat');
  opts = vl_argparse(opts, varargin) ;

  covFile = fullfile(opts.featDir, sprintf('%s-cov-est.mat', opts.model)) ;

  if exist(covFile, 'file') && opts.useCached, return ; end
  if exist(covFile, 'file')
    fprintf('cov file exists! are you sure??\n') ; pause() ; fprintf('ok \n') ;
  end

  net = load(fullfile(opts.modelDir, opts.model)) ;
  net = dagnn.DagNN.loadobj(net) ; net.mode = 'test';
  [net, tracked] = insert_cov_estimator_layers(net, opts) ;
  if numel(opts.gpus) > 0, gpuDevice(opts.gpus) ; net.move('gpu') ; end

  imdb = load(opts.imdbPath) ;
  samples = find(imdb.images.set == 1) ;
  samples = vl_colsubset(samples, opts.sampleSize) ;

  tic
  for ii = 1:numel(samples)
    batch = samples(ii) ;
    runNet(imdb, net, batch);
    if mod(ii,100)==1
      fprintf('(%d/%d): %.2f (Hz)\n', ii, numel(samples), ii/round(toc)) ;
      for li=1:numel(tracked)
        l_ = net.layers(net.getLayerIndex(tracked{li})).block ;
        df_ = l_.average ;
        mu_ = mean(l_.M12(:)) ;
        fprintf('%s, df = %1.2g, mn = %1.2g\n', tracked{li}, df_, mu_) ;
      end
      tic ;
    end
  end

  mus = {} ; covs = {} ;
  for ii=1:numel(tracked)
    layer = net.layers(net.getLayerIndex(tracked{ii}));
    mus{end+1} = gather(layer.block.mu) ; %#ok
    covs{end+1} = gather(layer.block.cov) ; %#ok
  end 

  VARS = opts.targetFeats ; %#ok
  if ~exist(fileparts(covFile), 'dir'), mkdir(fileparts(covFile)) ; end
  save(covFile, 'VARS', 'mus', 'covs') ;
  fprintf('saved statistics to %s\n', covFile) ;

% ------------------------------
function runNet(imdb, net, batch)
% ------------------------------
  imName = imdb.images.name{batch} ;
  rgbPath = fullfile(imdb.imageDir, imName) ;
  im = single(imread(rgbPath)) ;
  imClip = applyAugmentation(im,net.meta.normalization.imageSize(1:2)); 
  im_ = single(imClip) ;
  if size(im_,3) == 1, im_ = repmat(im_,[1 1 3]) ; end % handle greyscale
  avgIm = net.meta.normalization.averageImage ;
  if ~isempty(avgIm), im_ = im_ - avgIm ; end
  if strcmp(net.device, 'gpu'), im_ = gpuArray(im_) ; end
  net.eval({'data', im_}) ;

% -------------------------------------------
function imt = applyAugmentation(imt, imsize) 
% -------------------------------------------
%APPLYAUGMENTATION(IMT, IMSIZE) - apply standard image augmentation

  w = size(imt,2) ; h = size(imt,1) ;
  factor = [imsize(1)/h imsize(2)/w];
  factor = max(factor) ;

  if any(abs(factor - 1) > 0.0001)
    imt = imresize(imt, 'scale', factor, 'method','bicubic') ;
  end 

  % crop & flip
  w = size(imt,2) ; h = size(imt,1) ;
  sz = round(min(imsize(1:2)' .* (1-0.03+0.06*rand(2,1)), [h;w])) ;
  dx = randi(w - sz(2) + 1, 1) ; dy = randi(h - sz(1) + 1, 1) ;
  sx = round(linspace(dx, sz(2)+dx-1, imsize(2))) ;
  sy = round(linspace(dy, sz(1)+dy-1, imsize(1))) ;
  imt = imt(sy,sx,:) ;

% --------------------------------------------------------------
function [dag, tracked] = insert_cov_estimator_layers(dag, opts)
% --------------------------------------------------------------

  tracked = cell(1, numel(opts.targetFeats)) ;
  for ii = 1:numel(opts.targetFeats)
    idx = dag.getLayerIndex(opts.targetFeats{ii}) ;
    player = dag.layers(idx) ;
    name = sprintf('%s_cov_est', player.outputs{1}) ;
    outVar = {name} ; inputs = player.outputs ;
    block = dagnn.CovarianceEstimator() ;
    params = {[name '_mu'], [name '_sig']} ;
    dag.addLayer(name, block, inputs, outVar, params) ;
    dag.vars(dag.getVarIndex(outVar)).precious = 1 ;
    tracked{ii} = name ;
  end
  dag.rebuild() ;
