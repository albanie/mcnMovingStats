classdef CovarianceEstimator < dagnn.ElementWise
  
  properties(Transient)
    N = 0
    K = []
    mu = 0
    mu1 = 0
    M12 = 0
    cov = 0
    average = 0 
    updateDiff = 0 
    numAveraged = 0
    descCheck = []
  end

  methods
    function outputs = forward(obj, inputs, params)

      prevCov = obj.cov ;
      x_ = permute(inputs{1},[3 1 2 4]) ;
      x_ = reshape(x_, size(x_, 1), []) ;
      x_ = vl_colsubset(x_, 1e4) ; % restrict sample size for estimate if needed
      if isempty(obj.K), obj.K = mean(x_(:)) ; end % initialise mean estimate
      obj.N = obj.N + size(x_, 2) ; % total number of samples seen so far
      m12_ = (x_- obj.K) * (x_ - obj.K)' ; % covariance estimate from current batch
      obj.M12 = obj.M12 + m12_ ; % update covariance estimate
      obj.mu1 = obj.mu1 + sum(x_ - obj.K, 2) ; % update distance to 
      
      % if mean(obj.M12(:)) > 1e10
      %   fprintf('!! inaccurate ... !!\n') ;
      % end

      obj.cov = (obj.M12 - obj.mu1*obj.mu1'*(1/obj.N))*(1/(obj.N-1)) ;
      obj.mu = obj.mu1*(1 / obj.N) + obj.K ;

      % if true
      %   obj.descCheck  = [obj.descCheck vl_colsubset(x_,100)] ;
      %   disp(size(obj.descCheck)) ;
      %   if size(obj.descCheck,2) > 1e4
      %     xc_ = bsxfun(@minus,obj.descCheck,mean(obj.descCheck,2)) ;
      %     cv__ = xc_*xc_'/size(xc_,2) ;
      %     figure(1) ; clf ; 
      %     subplot(2,3,1) ; imagesc(gather(obj.cov)) ; colorbar ; 
      %     subplot(2,3,2) ; imagesc(gather(cv__)) ; colorbar ; 
      %     subplot(2,3,3) ; imagesc(gather(abs(cv__-obj.cov))) ; colorbar ;
      %     subplot(2,3,4) ; imagesc(gather(obj.mu)) ; colorbar ; 
      %     subplot(2,3,5) ; imagesc(gather(mean(obj.descCheck,2))) ; colorbar ; 
      %     subplot(2,3,6) ; imagesc(abs(gather(obj.mu-mean(obj.descCheck,2)))) ; colorbar ;
      %     saveas(1,'~/junk/covtest.png')
      %     keyboard
      %   end
      % end

      % Use Frobenius to track changes to Covariance matrixj
      df = norm(obj.cov(:)- prevCov(:), 'fro') ;
      obj.updateDiff = df ;

      n_ = obj.numAveraged  ;
      m_ = n_ + 1  ;
      obj.average = (n_ * obj.average + df) / m_  ;
      obj.numAveraged = m_  ;
      outputs = {obj.mu, obj.cov} ;
    end

    function reset(obj)
      obj.mu1 = 0  ;
      obj.M12 = 0  ;
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
