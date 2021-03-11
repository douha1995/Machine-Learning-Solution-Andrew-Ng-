function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = .3;

c =[.1;.2 ;.4 ;.8 ; 1 ;1.6 ;3.5 ;9.2 ;10];
Sigma =[.1 ;.3  ;.5 ;.7 ;.9 ;1.5 ;3 ;4 ;5];
minE = 1;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
  for i = 1 : size(c)
    for j = 1 :size(Sigma)
     SVMmodel = svmTrain(X,y,c(i),@(x1,x2)gaussianKernel(x1,x2,Sigma(j)));
     visualizeBoundaryLinear(X, y, SVMmodel);
     predictions = svmPredict(SVMmodel, Xval);
     e = mean(double(predictions ~= yval));
     if ( e < minE)
         minE = e;
         C = c(i);
         sigma = Sigma(j);
     end;
    end;
  end;



e
% =========================================================================

end
