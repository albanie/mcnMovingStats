function dag = insert_cov_estimator_layers(dag, opts)

  idx = dag.getLayerIndex(opts.targetFeats) ;
  for ii = 1:numel(idx)
    player = dag.layers(idx) ;
    name = sprintf('%s_cov_est', player.outputs{1}) ;
    outVar = {name} ; inputs = player.outputs ;
    block = dagnn.CovarianceEstimator() ;
    dag.addLayer(name, block, inputs, outVar, {}) ;
  end
  dag.rebuild() ;
