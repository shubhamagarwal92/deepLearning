function [ E ] = energy( W,U,biasVisInput,biasVisOutput,biasHidden,dataX, dataY )

E = 0;
for i = 1:size(dataX,1)
   x = dataX(i,:)';
   y = zeros(10,1);
   y(dataY(i)+1) = 1;
   activationHidden = biasHidden + U(:,dataY(i)+1) + W*x;
   hProb = sigmoid(activationHidden);
   randhProb = rand(size(hProb,1),1);
   h = double(hProb > randhProb);
end
E = E + -h'*W*x - biasVisInput'*x - biasVisOutput'*y - biasHidden'*h - h'*U*y;

end

