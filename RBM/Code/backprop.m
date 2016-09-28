function [W,U] = backprop(W,U,visState,targetVal)
        delW=zeros(size(W));
        delU=zeros(size(U));
        numHidden=size(U,1);
        numVisOut=size(U,2);
        learnRate=0.01;
        netHid=zeros(numHidden,1); %%%column 
        netVisOut=zeros(numVisOut,1);
        netHid=W*visState;
        outHid=sigmoid(netHid); %HX1
        netVisOut=U'*outHid;
        outVisOut=sigmoid(netVisOut);
        errorOut=(targetVal-outVisOut).*outVisOut.*(1-outVisOut); %Yx1
        delU=learn*outHid.*errorOut'; 
        deltasum=U*errorOut; %Hx1
        delta=outHid.*(1-outHid).*deltasum; %Hx1
        delW=learn*delta*visState; %HxV
        W=W+delW;
        U=U+delU;        

end
