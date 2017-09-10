function setup_mcnMovingStats()
%SETUP_MCNMOVINGSTATS Sets up mcnMovingStats, by adding its folders 
% to the Matlab path
%
% Copyright (C) 2017 Samuel Albanie and David Novotny
% Licensed under The MIT License [see LICENSE.md for details]

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/imagenet']) ;
  addpath([vl_rootnn '/examples/imagenet'], [vl_rootnn, '/examples']) ;
