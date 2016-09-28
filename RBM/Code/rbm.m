clear all;
trainX = loadMNISTImages('../Datasets/MNIST/train-images.idx3-ubyte')';
trainY = loadMNISTLabels('../Datasets/MNIST/train-labels.idx1-ubyte');

testX = loadMNISTImages('../Datasets/MNIST/t10k-images.idx3-ubyte')';
testY = loadMNISTLabels('../Datasets/MNIST/t10k-labels.idx1-ubyte');

threshold = 0.1;
trainX = double(trainX > threshold);
testX = double(testX > threshold);
numVisInput = size(trainX,2);
numVisOutput = 10;
numHidden = 100;
lRate = 0.0005;
W = rand(numHidden,numVisInput);
U = rand(numHidden,numVisOutput);
biasVisInput = ones(numVisInput,1);
biasVisOutput = ones(numVisOutput,1);
biasHidden = ones(numHidden,1);

% 
% im = trainX(1,:);
% im = im>0.1;
% im1 = zeros(28,28);
% for i=1:28
%     for j=1:28
%         im1(j,i) = im((i-1)*28+j);
%     end
% end
% imshow(im1);
% for i=1:size(trainX,1)

E = energy(W,U,biasVisInput,biasVisOutput,biasHidden,trainX(1:100,:),trainY(1:100,:));
disp(sprintf('Initial Energy = %d',E));
% TRAINING VIA MAXIMUM LIKELIHOOD

for epoch = 1:1
    for i = 1:10000
       x1 = trainX(i,:)';
       y1 = zeros(10,1);
       y1(trainY(i)+1) = 1;
    %    POSITIVE STEPS
       activationHidden1 = biasHidden + U(:,trainY(i)+1) + W*x1;
       h1Prob = sigmoid(activationHidden1);
       randh1Prob = rand(size(h1Prob,1),1);
       h1 = double(h1Prob > randh1Prob);
    %    NEGATIVE STEPS
       activationVisOutput =  biasVisOutput + U'*h1;
       [M,I] = max(activationVisOutput);
       y2 = zeros(10,1);
       y2(I)= 1;
       activationVisInput = biasVisInput + W'*h1;
       x2Prob = sigmoid(activationVisInput);
       randx2Prob = rand(size(x2Prob));
       x2 = double(x2Prob > randx2Prob);
       activationHidden2 = biasHidden + U(:,I) + W*x2;
       h2Prob = sigmoid(activationHidden2);
       randh2Prob = rand(size(h2Prob,1),1);
       h2 = double(h2Prob > randh2Prob);
    %    UPDATE STEPS
       W = W + lRate*(h1Prob*x1' - h2Prob*x2');
       U = U + lRate*(h1Prob*y1' - h2Prob*y2');
       biasHidden = biasHidden + lRate*(h1Prob - h2Prob);
       biasVisInput = biasVisInput + lRate*(x1 - x2);
       biasVisOutput = biasVisOutput + lRate*(y1 - y2);
    end
    E = energy(W,U,biasVisInput,biasVisOutput,biasHidden,trainX(1:10000,:),trainY(1:10000,:));
    disp(sprintf('After epoch %d, Energy = %d',epoch,E));

end

% CLASSIFICATION
numCorrect = 0;
for i=1:size(testX,1)
   x = testX(i,:)';
   yProb = zeros(10,1); 
   for j=1:10
       activationTerm = biasHidden + U(:,j) + W*x;
       expTerm = exp(activationTerm);
       sumTerm = 1 + expTerm;
       productTerm = prod(sumTerm);
       yProb(j) = productTerm*exp(biasVisOutput(j));
   end
   [M,I] = max(yProb);
   if(testY(i) == (I-1))
       numCorrect = numCorrect + 1;
   end
end

accuracy = numCorrect/size(testX,1)*100.0


