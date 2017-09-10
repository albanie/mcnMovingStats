classdef MovingStats < dagnn.ElementWise
  
  properties(Transient)
    N = 0
    K = []
    mu = 0
    sm1 = 0 % shifted first moment
    sm2 = 0 % shifted second moment
    cov = 0
    average = 0 
    updateDiff = 0 
    numAveraged = 0
    descCheck = []
    vis = 0 
  end

  methods
    function outputs = forward(obj, inputs, params)

      prevCov = obj.cov ;

      % provide an offset to ensure numerical stability (initisalise to mean)
      if isempty(obj.K), obj.K = mean(inputs{1}(:)) ; end 
      [obj.sm1, obj.sm2, obj.mu, obj.cov, obj.N] ...
            = vl_nnmovingstats(inputs{1}, obj.sm1, obj.sm2, obj.N, 'K', obj.K) ;

       if obj.vis % optional visualisation code
         obj.descCheck  = [obj.descCheck vl_colsubset(x_,100)] ;
         disp(size(obj.descCheck)) ;
         if size(obj.descCheck,2) > 1e4
           xc_ = bsxfun(@minus,obj.descCheck,mean(obj.descCheck,2)) ;
           cv__ = xc_*xc_'/size(xc_,2) ;
           figure(1) ; clf ; 
           subplot(2,3,1) ; imagesc(gather(obj.cov)) ; colorbar ; 
           subplot(2,3,2) ; imagesc(gather(cv__)) ; colorbar ; 
           subplot(2,3,3) ; imagesc(gather(abs(cv__-obj.cov))) ; colorbar ;
           subplot(2,3,4) ; imagesc(gather(obj.mu)) ; colorbar ; 
           subplot(2,3,5) ; imagesc(gather(mean(obj.descCheck,2))) ; colorbar ; 
           subplot(2,3,6) ; imagesc(abs(gather(obj.mu-mean(obj.descCheck,2)))) ; colorbar ;
           saveas(1,'~/junk/covtest.png')
           keyboard
         end
       end

      % Use Frobenius to track changes to Covariance matrix
      df = norm(obj.cov(:) - prevCov(:), 'fro') ;
      obj.updateDiff = df ;
      n_ = obj.numAveraged  ;
      m_ = n_ + 1  ;
      obj.average = (n_ * obj.average + df) / m_  ;
      obj.numAveraged = m_  ;
      outputs = {obj.mu, obj.cov} ;
    end

    function reset(obj)
      obj.sm1 = 0  ;
      obj.sm2 = 0  ;
      obj.mu = 0  ;
      obj.cov = 0  ;
      obj.K = [] ;
      obj.updateDiff = 0 ;
      obj.average = 0  ;
      obj.numAveraged = 0 ;
      obj.N = 0 ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = 0;
      derParams = [];
    end

    function obj = CovarianceEstBatchAcc(varargin)
      obj.load(varargin{:})  ;
    end


    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      for ii=1:2
        outputSizes{ii} = NaN*ones(1,4)  ;
      end
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      for ii=1
        for jj=1:2 
          rfs(ii,jj).size = [NaN NaN]  ;
          rfs(ii,jj).stride = [NaN NaN]  ;
          rfs(ii,jj).offset = [NaN NaN]  ; 
        end
      end 
    end
  end
end
