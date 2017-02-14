function [data1, data2]=CGMOS(gdata1, gdata2, newDataRatio, sigmafactor=1.0)
    featureImportance=ones(1, size(gdata1, 2));
    dSize1=size(gdata1, 1);
    dSize2=size(gdata2, 1);
    newDataNum=ceil(abs(dSize1-dSize2)*newDataRatio);
    w=1.0;
    % For balanced maj and min data
%     w=(dSize1-dSize2+newDataNum)/(2*newDataNum);
    if isnan(w)
        w=1.0;
    end
    [data1, data2]=Upsampling(gdata1, gdata2, featureImportance, sigmafactor, newDataNum, w);
end

function [data1, data2]=Upsampling(gdata1, gdata2, featureImportance, sigmafactor, newDataNum, w)
% gdata1: majority class
% gdata2: minority class
k=10;

dSize1=size(gdata1, 1);
dSize2=size(gdata2, 1);

batchNum=uint16(newDataNum);
featureSelectionRounds=1;

for n=1:featureSelectionRounds    
    fSize=size(gdata1, 2);
    ratio=1.0;
    fNum=max(1, floor(fSize*ratio));
    fIdx=randperm(fSize, fNum);
    tmpdata1=gdata1(:, fIdx);
    tmpdata2=gdata2(:, fIdx);

    [knnIdx1, d1]=knnsearch(gather(tmpdata1), gather(tmpdata1), 'k', k);
    [knnIdx2, d2]=knnsearch(gather(tmpdata2), gather(tmpdata2), 'k', k);

    gsigma1=mean((d1'))*sigmafactor;
    gsigma2=mean((d2'))*sigmafactor;
    [gpdfOfPts_D1_K1, gdensity_D1_K1]=GetPdfOfPoints(gsigma1, tmpdata1, tmpdata1);
    [gpdfOfPts_D2_K1, gdensity_D2_K1]=GetPdfOfPoints(gsigma1, tmpdata1, tmpdata2);
    [gpdfOfPts_D1_K2, gdensity_D1_K2]=GetPdfOfPoints(gsigma2, tmpdata2, tmpdata1);
    [gpdfOfPts_D2_K2, gdensity_D2_K2]=GetPdfOfPoints(gsigma2, tmpdata2, tmpdata2);

    gconffidence_D1=GetConffidence(gpdfOfPts_D1_K1, gpdfOfPts_D1_K2, dSize1, dSize2);
    gconffidence_D2=GetConffidence(gpdfOfPts_D2_K2, gpdfOfPts_D2_K1, dSize2, dSize1);

    %% Search for seed in data1
    %  Compute conffidence ratio for data1 upon new data added to data1
    gpdfOfPts_D1_K1_Mat=repmat(gpdfOfPts_D1_K1, 1, dSize1);
    gpdfOfPts_D1_K1_Mat=(gpdfOfPts_D1_K1_Mat*dSize1+gdensity_D1_K1)/(dSize1+1);
    gpdfOfPts_D1_K1_Mat=[gpdfOfPts_D1_K1_Mat; diag(gpdfOfPts_D1_K1_Mat)']; % include candidates
    gpdfOfPts_D1_K2_Mat=repmat(gpdfOfPts_D1_K2, 1, dSize1);
    gpdfOfPts_D1_K2_Mat=[gpdfOfPts_D1_K2_Mat; diag(gpdfOfPts_D1_K2_Mat)']; % include candidates
    gconffidence_D1_New=GetConffidence(gpdfOfPts_D1_K1_Mat, gpdfOfPts_D1_K2_Mat, dSize1+1, dSize2);
    % Compute conffidence ratio for data2 upon new data added to data1
    gpdfOfPts_D2_K1_Mat=repmat(gpdfOfPts_D2_K1, 1, dSize1);
    gpdfOfPts_D2_K1_Mat=(gpdfOfPts_D2_K1_Mat*dSize1+gdensity_D2_K1)/(dSize1+1);
    gpdfOfPts_D2_K2_Mat=repmat(gpdfOfPts_D2_K2, 1, dSize1);    
    gconffidence_D2_New=GetConffidence(gpdfOfPts_D2_K2_Mat, gpdfOfPts_D2_K1_Mat, dSize2, dSize1+1);

    gconffidence_NewInD1=([gconffidence_D1_New; gconffidence_D2_New]);
    gconffidence_OldInD1=([[repmat(gconffidence_D1, 1, dSize1); gconffidence_D1']; repmat(gconffidence_D2, 1, dSize1)]);    
    gconffidence_Ratio_D1=(gconffidence_NewInD1-gconffidence_OldInD1)./gconffidence_OldInD1;
    gconffidence_Ratio_D1=0.5*(mean(gconffidence_Ratio_D1(1:dSize1+1, :))+mean(gconffidence_Ratio_D1(dSize1+2:end, :)));

    %% Search for seed in data2
    % Compute conffidence ratio for data2 upon new data added to data2
    gpdfOfPts_D2_K2_Mat=repmat(gpdfOfPts_D2_K2, 1, dSize2);
    gpdfOfPts_D2_K2_Mat=(gpdfOfPts_D2_K2_Mat*dSize2+gdensity_D2_K2)/(dSize2+1);
    gpdfOfPts_D2_K2_Mat=[gpdfOfPts_D2_K2_Mat; diag(gpdfOfPts_D2_K2_Mat)']; % include candidates
    gpdfOfPts_D2_K1_Mat=repmat(gpdfOfPts_D2_K1, 1, dSize2); 
    gpdfOfPts_D2_K1_Mat=[gpdfOfPts_D2_K1_Mat; diag(gpdfOfPts_D2_K1_Mat)']; % include candidates
    gconffidence_D2_New=GetConffidence(gpdfOfPts_D2_K2_Mat, gpdfOfPts_D2_K1_Mat, dSize2+1, dSize1);
    % Compute conffidence ratio for data2 upon new data added to data2
    gpdfOfPts_D1_K2_Mat=repmat(gpdfOfPts_D1_K2, 1, dSize2);
    gpdfOfPts_D1_K2_Mat=(gpdfOfPts_D1_K2_Mat*dSize2+gdensity_D1_K2)/(dSize2+1);
    gpdfOfPts_D1_K1_Mat=repmat(gpdfOfPts_D1_K1, 1, dSize2);
    gconffidence_D1_New=GetConffidence(gpdfOfPts_D1_K1_Mat, gpdfOfPts_D1_K2_Mat, dSize1, dSize2+1);

    gconffidence_NewInD2=([gconffidence_D1_New; gconffidence_D2_New]);
    gconffidence_OldInD2=([repmat(gconffidence_D1, 1, dSize2); [repmat(gconffidence_D2, 1, dSize2); gconffidence_D2']]);
    gconffidence_Ratio_D2=(gconffidence_NewInD2-gconffidence_OldInD2)./gconffidence_OldInD2;
    gconffidence_Ratio_D2=0.5*(mean(gconffidence_Ratio_D2(1:dSize1, :))+mean(gconffidence_Ratio_D2(dSize1+1:end, :)));

    gconffidenceBatch(n, :)=[gconffidence_Ratio_D1, gconffidence_Ratio_D2];
    gsigmaBatch1(n, :)=gsigma1;
    gsigmaBatch2(n, :)=gsigma2;
end

    if size(gconffidenceBatch, 1)~=1
        gconffidence=mean(gconffidenceBatch);
        gsigma1=mean(gsigmaBatch1);
        gsigma2=mean(gsigmaBatch2);
    else
        gconffidence=(gconffidenceBatch);
        gsigma1=(gsigmaBatch1);
        gsigma2=(gsigmaBatch2);        
    end

    [gdata1, gdata2, gsigma1, gsigma2]=GetNewDataByInterpolationRandomSimplex3(gdata1, gdata2, gsigma1, gsigma2, gconffidence, batchNum, w);
    %%
    data1=gather(gdata1);
    data2=gather(gdata2);
end

function [gdata1, gdata2, gsigma1, gsigma2]=GetNewDataByInterpolationRandom(gdata1, gdata2, gsigma1, gsigma2, gconffidence, batchNum, w)
    k=min(5, size(gdata2, 1)-1);
    gdSize1=size(gdata1, 1);
    gdSize2=size(gdata2, 1);    
    gconffidence=(gconffidence-min(gconffidence))/(max(gconffidence)-min(gconffidence));
    
    if sum(gconffidence)==0
        gconffidence=zeros(size(gconffidence))+1;
    end
    
    classWeight=[1-w, w];
    class=randsample([1:2],batchNum,true,gather(classWeight));
    
    batchNum1=sum(class==1); %%%%%%%%%%%%%%%%%%%%%%%%
%     batchNum1=ceil(batchNum*0.5);
    if batchNum1~=0
        gconffidence1=gconffidence(1:gdSize1);
        gconffidence1=(gconffidence1-min(gconffidence1))/(max(gconffidence1)-min(gconffidence1));
        gconffidence1=gconffidence1.^1;
        [tmp, idx]=sort(gconffidence1);
        i=uint16(gdSize1*1/gdSize1);
        threshold=tmp(i);
        gconffidence1(gconffidence1<threshold)=0;
        
        if sum(gconffidence1)==0
            gconffidence1=ones(size(gconffidence1));
        end
        
        if numel(find(isnan(gconffidence1)))~=0 || numel(find(isinf(gconffidence1)))~=0
            gconffidence1=ones(size(gconffidence1));
        end
        
        maxConffidenceIdx=randsample([1:gdSize1],batchNum1,true,gather(gconffidence1));
        poiPos=gdata1(maxConffidenceIdx, :);
        poiSigma=gsigma1(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata1), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum1)));
        nbPos=gdata1(nbIdx, :);
        nbSigma=gsigma1(nbIdx);
        w=rand(batchNum1, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata1, 2))+nbPos.*(1-repmat(w, 1, size(gdata1, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma1=[gsigma1, newSigma];
        gdata1=[gdata1; newDataPos];
    end
    
    batchNum2=sum(class==2);
    if batchNum2~=0
        gconffidence2=gconffidence(gdSize1+1:end);
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        gconffidence2=gconffidence2.^1;%%%%%%%%%%%%%%%%%%%%%%%% Normally apply 1.
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        [tmp, idx]=sort(gconffidence2);
        i=uint16(gdSize2*1/gdSize2);
        threshold=tmp(i);
        gconffidence2(gconffidence2<threshold)=0;
        
        if sum(gconffidence2)==0
            gconffidence2=ones(size(gconffidence2));
        end
        
        if numel(find(isnan(gconffidence2)))~=0 || numel(find(isinf(gconffidence2)))~=0
            gconffidence2=ones(size(gconffidence2));
        end
        
        maxConffidenceIdx=randsample([1:gdSize2],batchNum2,true,gather(gconffidence2));
        poiPos=gdata2(maxConffidenceIdx, :);
        poiSigma=gsigma2(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata2), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum2)));
        nbPos=gdata2(nbIdx, :);
        nbSigma=gsigma2(nbIdx);
        w=rand(batchNum2, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata2, 2))+nbPos.*(1-repmat(w, 1, size(gdata2, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma2=[gsigma2, newSigma];
        gdata2=[gdata2; newDataPos];
    end
end

function [gdata1, gdata2, gsigma1, gsigma2]=GetNewDataByInterpolationWeightedRandom(gdata1, gdata2, gsigma1, gsigma2, gconffidence, batchNum, w)
    k=min(5, size(gdata2, 1)-1);
    gdSize1=size(gdata1, 1);
    gdSize2=size(gdata2, 1);    
    gconffidence=(gconffidence-min(gconffidence))/(max(gconffidence)-min(gconffidence));
    
    if sum(gconffidence)==0
        gconffidence=zeros(size(gconffidence))+1;
    end
    
    classWeight=[1-w, w];
    class=randsample([1:2],batchNum,true,gather(classWeight));
    
    batchNum1=sum(class==1); %%%%%%%%%%%%%%%%%%%%%%%%
%     batchNum1=ceil(batchNum*0.5);
    if batchNum1~=0
        gconffidence1=gconffidence(1:gdSize1);
        gconffidence1=(gconffidence1-min(gconffidence1))/(max(gconffidence1)-min(gconffidence1));
        gconffidence1=gconffidence1.^1;
        [tmp, idx]=sort(gconffidence1);
        i=uint16(gdSize1*1/gdSize1);
        threshold=tmp(i);
        gconffidence1(gconffidence1<threshold)=0;
        
        if sum(gconffidence1)==0
            gconffidence1=ones(size(gconffidence1));
        end
        
        if numel(find(isnan(gconffidence1)))~=0 || numel(find(isinf(gconffidence1)))~=0
            gconffidence1=ones(size(gconffidence1));
        end
        
        maxConffidenceIdx=randsample([1:gdSize1],batchNum1,true,gather(gconffidence1));
        poiPos=gdata1(maxConffidenceIdx, :);
        poiSigma=gsigma1(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata1), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        d(:, 1)=[];
        
        weight=1./d;
        for i=1:batchNum2
            w = weight(i, :);
            nbIdx(i) = knnIdx(i, randsample([1:k], 1, true, w));           
        end
        
        nbPos=gdata1(nbIdx, :);
        nbSigma=gsigma1(nbIdx);
        w=rand(batchNum1, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata1, 2))+nbPos.*(1-repmat(w, 1, size(gdata1, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma1=[gsigma1, newSigma];
        gdata1=[gdata1; newDataPos];
    end
    
    batchNum2=sum(class==2);
    if batchNum2~=0
        gconffidence2=gconffidence(gdSize1+1:end);
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        gconffidence2=gconffidence2.^1;%%%%%%%%%%%%%%%%%%%%%%%% Normally apply 1.
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        [tmp, idx]=sort(gconffidence2);
        i=uint16(gdSize2*1/gdSize2);
        threshold=tmp(i);
        gconffidence2(gconffidence2<threshold)=0;
        
        if sum(gconffidence2)==0
            gconffidence2=ones(size(gconffidence2));
        end
        
        if numel(find(isnan(gconffidence2)))~=0 || numel(find(isinf(gconffidence2)))~=0
            gconffidence2=ones(size(gconffidence2));
        end
        
        maxConffidenceIdx=randsample([1:gdSize2],batchNum2,true,gather(gconffidence2));
        poiPos=gdata2(maxConffidenceIdx, :);
        poiSigma=gsigma2(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata2), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        d(:, 1)=[];
        
        weight=1./d;
        for i=1:batchNum2
            w = weight(i, :);
            nbIdx(i) = knnIdx(i, randsample([1:k], 1, true, w));           
        end

        nbPos=gdata2(nbIdx, :);
        nbSigma=gsigma2(nbIdx);
        w=rand(batchNum2, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata2, 2))+nbPos.*(1-repmat(w, 1, size(gdata2, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma2=[gsigma2, newSigma];
        gdata2=[gdata2; newDataPos];
    end
end

function [gdata1, gdata2, gsigma1, gsigma2]=GetNewDataByInterpolationRandomTriangle(gdata1, gdata2, gsigma1, gsigma2, gconffidence, batchNum, w)
    k=min(5, size(gdata2, 1)-1);
    gdSize1=size(gdata1, 1);
    gdSize2=size(gdata2, 1);    
    gconffidence=(gconffidence-min(gconffidence))/(max(gconffidence)-min(gconffidence));
    
    if sum(gconffidence)==0
        gconffidence=zeros(size(gconffidence))+1;
    end
    
    classWeight=[1-w, w];
    class=randsample([1:2],batchNum,true,gather(classWeight));
    
    batchNum1=sum(class==1); %%%%%%%%%%%%%%%%%%%%%%%%
%     batchNum1=ceil(batchNum*0.5);
    if batchNum1~=0
        gconffidence1=gconffidence(1:gdSize1);
        gconffidence1=(gconffidence1-min(gconffidence1))/(max(gconffidence1)-min(gconffidence1));
        gconffidence1=gconffidence1.^1;
        [tmp, idx]=sort(gconffidence1);
        i=uint16(gdSize1*1/gdSize1);
        threshold=tmp(i);
        gconffidence1(gconffidence1<threshold)=0;
        
        if sum(gconffidence1)==0
            gconffidence1=ones(size(gconffidence1));
        end
        
        if numel(find(isnan(gconffidence1)))~=0 || numel(find(isinf(gconffidence1)))~=0
            gconffidence1=ones(size(gconffidence1));
        end
        
        maxConffidenceIdx=randsample([1:gdSize1],batchNum1,true,gather(gconffidence1));
        poiPos=gdata1(maxConffidenceIdx, :);
        poiSigma=gsigma1(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata1), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum1)));
        nbPos=gdata1(nbIdx, :);
        nbSigma=gsigma1(nbIdx);
        w=rand(batchNum1, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata1, 2))+nbPos.*(1-repmat(w, 1, size(gdata1, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma1=[gsigma1, newSigma];
        gdata1=[gdata1; newDataPos];
    end
    
    batchNum2=sum(class==2);
    if batchNum2~=0
        gconffidence2=gconffidence(gdSize1+1:end);
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        gconffidence2=gconffidence2.^1;%%%%%%%%%%%%%%%%%%%%%%%% Normally apply 1.
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        [tmp, idx]=sort(gconffidence2);
        i=uint16(gdSize2*1/gdSize2);
        threshold=tmp(i);
        gconffidence2(gconffidence2<threshold)=0;
        
        if sum(gconffidence2)==0
            gconffidence2=ones(size(gconffidence2));
        end
        
        if numel(find(isnan(gconffidence2)))~=0 || numel(find(isinf(gconffidence2)))~=0
            gconffidence2=ones(size(gconffidence2));
        end
        
        maxConffidenceIdx=randsample([1:gdSize2],batchNum2,true,gather(gconffidence2));
        poiPos=gdata2(maxConffidenceIdx, :);
        poiSigma=gsigma2(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata2), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum2)));
        nbPos1=gdata2(nbIdx, :);
        nbSigma1=gsigma2(nbIdx);
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum2)));
        nbPos2=gdata2(nbIdx, :);
        nbSigma2=gsigma2(nbIdx);
        w1=repmat(rand(batchNum2, 1), 1, size(gdata2, 2));
        w2=repmat(rand(batchNum2, 1), 1, size(gdata2, 2));
        
        newDataPos=(1-sqrt(w1)).*poiPos+(sqrt(w1).*(1-w2)).*nbPos1+(w2.*sqrt(w1)).*nbPos2;
        w1=w1(:, 1)';
        w2=w2(:, 1)';
        newSigma=(1-sqrt(w1)).*poiSigma+(sqrt(w1).*(1-w2)).*nbSigma1+(w2.*sqrt(w1)).*nbSigma2;
        gsigma2=[gsigma2, newSigma];
        gdata2=[gdata2; newDataPos];
    end
end

function [gdata1, gdata2, gsigma1, gsigma2]=GetNewDataByInterpolationRandomWeightedTriangle(gdata1, gdata2, gsigma1, gsigma2, gconffidence, batchNum, w)
    k=min(5, size(gdata2, 1)-1);
    gdSize1=size(gdata1, 1);
    gdSize2=size(gdata2, 1);    
    gconffidence=(gconffidence-min(gconffidence))/(max(gconffidence)-min(gconffidence));
    
    if sum(gconffidence)==0
        gconffidence=zeros(size(gconffidence))+1;
    end
    
    classWeight=[1-w, w];
    class=randsample([1:2],batchNum,true,gather(classWeight));
    
    batchNum1=sum(class==1); %%%%%%%%%%%%%%%%%%%%%%%%
%     batchNum1=ceil(batchNum*0.5);
    if batchNum1~=0
        gconffidence1=gconffidence(1:gdSize1);
        gconffidence1=(gconffidence1-min(gconffidence1))/(max(gconffidence1)-min(gconffidence1));
        gconffidence1=gconffidence1.^1;
        [tmp, idx]=sort(gconffidence1);
        i=uint16(gdSize1*1/gdSize1);
        threshold=tmp(i);
        gconffidence1(gconffidence1<threshold)=0;
        
        if sum(gconffidence1)==0
            gconffidence1=ones(size(gconffidence1));
        end
        
        if numel(find(isnan(gconffidence1)))~=0 || numel(find(isinf(gconffidence1)))~=0
            gconffidence1=ones(size(gconffidence1));
        end
        
        maxConffidenceIdx=randsample([1:gdSize1],batchNum1,true,gather(gconffidence1));
        poiPos=gdata1(maxConffidenceIdx, :);
        poiSigma=gsigma1(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata1), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum1)));
        nbPos=gdata1(nbIdx, :);
        nbSigma=gsigma1(nbIdx);
        w=rand(batchNum1, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata1, 2))+nbPos.*(1-repmat(w, 1, size(gdata1, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma1=[gsigma1, newSigma];
        gdata1=[gdata1; newDataPos];
    end
    
    batchNum2=sum(class==2);
    if batchNum2~=0
        gconffidence2=gconffidence(gdSize1+1:end);
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        gconffidence2=gconffidence2.^1;%%%%%%%%%%%%%%%%%%%%%%%% Normally apply 1.
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        [tmp, idx]=sort(gconffidence2);
        i=uint16(gdSize2*1/gdSize2);
        threshold=tmp(i);
        gconffidence2(gconffidence2<threshold)=0;
        
        if sum(gconffidence2)==0
            gconffidence2=ones(size(gconffidence2));
        end
        
        if numel(find(isnan(gconffidence2)))~=0 || numel(find(isinf(gconffidence2)))~=0
            gconffidence2=ones(size(gconffidence2));
        end
        
        maxConffidenceIdx=randsample([1:gdSize2],batchNum2,true,gather(gconffidence2));
        poiPos=gdata2(maxConffidenceIdx, :);
        poiSigma=gsigma2(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata2), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum2)));
        nbPos1=gdata2(nbIdx, :);
        nbSigma1=gsigma2(nbIdx);
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum2)));
        nbPos2=gdata2(nbIdx, :);
        nbSigma2=gsigma2(nbIdx);
        w1=repmat(rand(batchNum2, 1), 1, size(gdata2, 2));
        w2=repmat(rand(batchNum2, 1), 1, size(gdata2, 2));
        
        meanPos = (poiPos + nbPos1 + nbPos2)/3;
        poiDist = diag(pdist2(poiPos, meanPos));
        nb1Dist = diag(pdist2(nbPos1, meanPos));
        nb2Dist = diag(pdist2(nbPos2, meanPos));
        meanDist = (poiDist + nb1Dist + nb2Dist)/3;
        meanDist(find(meanDist==0))=1;
        rpoi = meanDist./poiDist;
        rnb1 = meanDist./nb1Dist;
        rnb2 = meanDist./nb2Dist;
        rpoi(find(rpoi==0))=1;
        rnb1(find(rnb1==0))=1;
        rnb2(find(rnb2==0))=1;
        rpoi(find(isnan(rpoi)))=1;
        rnb1(find(isnan(rnb1)))=1;
        rnb2(find(isnan(rnb2)))=1;
        
        scale = 0.1; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
        tmprpoi = rpoi.^scale;
        tmprnb1 = rnb1.^scale;
        tmprnb2 = rnb2.^scale;
        
        rpoi = tmprpoi./(tmprpoi+tmprnb1+tmprnb2);
        rnb1 = tmprnb1./(tmprpoi+tmprnb1+tmprnb2);
        rnb2 = tmprnb2./(tmprpoi+tmprnb1+tmprnb2);
        
        rpoi(find(rpoi==0))=1;
        rnb1(find(rnb1==0))=1;
        rnb2(find(rnb2==0))=1;
        rpoi(find(isnan(rpoi)))=1;
        rnb1(find(isnan(rnb1)))=1;
        rnb2(find(isnan(rnb2)))=1;
        
        trpoi=rpoi;
        trnb1=rnb1;
        trnb2=rnb2;
        
        rpoi = trpoi./(trpoi+trnb1+trnb2);
        rnb1 = trnb1./(trpoi+trnb1+trnb2);
        rnb2 = trnb2./(trpoi+trnb1+trnb2);
        
        tmpweight1 = repmat(rpoi, 1, size(gdata2, 2)).*(1-sqrt(w1));
        tmpweight2 = repmat(rnb1, 1, size(gdata2, 2)).*(sqrt(w1).*(1-w2));
        tmpweight3 = repmat(rnb2, 1, size(gdata2, 2)).*(w2.*sqrt(w1));
     
        weight1 = tmpweight1./(tmpweight1 + tmpweight2 + tmpweight3);
        weight2 = tmpweight2./(tmpweight1 + tmpweight2 + tmpweight3);
        weight3 = tmpweight3./(tmpweight1 + tmpweight2 + tmpweight3);
        
        weight1(find(weight1==0))=1;
        weight2(find(weight2==0))=1;
        weight3(find(weight3==0))=1;
        weight1(find(isnan(weight1)))=1;
        weight2(find(isnan(weight2)))=1;
        weight3(find(isnan(weight3)))=1;
        
        tmpweight1 = weight1;
        tmpweight2 = weight2;
        tmpweight3 = weight3;
        
        weight1 = tmpweight1./(tmpweight1 + tmpweight2 + tmpweight3);
        weight2 = tmpweight2./(tmpweight1 + tmpweight2 + tmpweight3);
        weight3 = tmpweight3./(tmpweight1 + tmpweight2 + tmpweight3);
        
        newDataPos=weight1.*poiPos+weight2.*nbPos1+weight3.*nbPos2;
        weight1 = weight1(:, 1)';
        weight2 = weight2(:, 2)';
        weight3 = weight3(:, 3)';
        
        newSigma=weight1.*poiSigma+weight2.*nbSigma1+weight3.*nbSigma2;
        gsigma2=[gsigma2, newSigma];
        gdata2=[gdata2; newDataPos];
    end
end

function [gdata1, gdata2, gsigma1, gsigma2]=GetNewDataByInterpolationRandomSimplex(gdata1, gdata2, gsigma1, gsigma2, gconffidence, batchNum, w)
    k=min(5, size(gdata2, 1)-1);
    gdSize1=size(gdata1, 1);
    gdSize2=size(gdata2, 1);    
    gconffidence=(gconffidence-min(gconffidence))/(max(gconffidence)-min(gconffidence));
    
    if sum(gconffidence)==0
        gconffidence=zeros(size(gconffidence))+1;
    end
    
    classWeight=[1-w, w];
    class=randsample([1:2],batchNum,true,gather(classWeight));
    
    batchNum1=sum(class==1); %%%%%%%%%%%%%%%%%%%%%%%%
%     batchNum1=ceil(batchNum*0.5);
    if batchNum1~=0
        gconffidence1=gconffidence(1:gdSize1);
        gconffidence1=(gconffidence1-min(gconffidence1))/(max(gconffidence1)-min(gconffidence1));
        gconffidence1=gconffidence1.^1;
        [tmp, idx]=sort(gconffidence1);
        i=uint16(gdSize1*1/gdSize1);
        threshold=tmp(i);
        gconffidence1(gconffidence1<threshold)=0;
        
        if sum(gconffidence1)==0
            gconffidence1=ones(size(gconffidence1));
        end
        
        if numel(find(isnan(gconffidence1)))~=0 || numel(find(isinf(gconffidence1)))~=0
            gconffidence1=ones(size(gconffidence1));
        end
        
        maxConffidenceIdx=randsample([1:gdSize1],batchNum1,true,gather(gconffidence1));
        poiPos=gdata1(maxConffidenceIdx, :);
        poiSigma=gsigma1(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata1), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum1)));
        nbPos=gdata1(nbIdx, :);
        nbSigma=gsigma1(nbIdx);
        w=rand(batchNum1, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata1, 2))+nbPos.*(1-repmat(w, 1, size(gdata1, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma1=[gsigma1, newSigma];
        gdata1=[gdata1; newDataPos];
    end
    
    batchNum2=sum(class==2);
    if batchNum2~=0
        gconffidence2=gconffidence(gdSize1+1:end);
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        gconffidence2=gconffidence2.^1;%%%%%%%%%%%%%%%%%%%%%%%% Normally apply 1.
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        [tmp, idx]=sort(gconffidence2);
        i=uint16(gdSize2*1/gdSize2);
        threshold=tmp(i);
        gconffidence2(gconffidence2<threshold)=0;
        
        if sum(gconffidence2)==0
            gconffidence2=ones(size(gconffidence2));
        end
        
        if numel(find(isnan(gconffidence2)))~=0 || numel(find(isinf(gconffidence2)))~=0
            gconffidence2=ones(size(gconffidence2));
        end
        
        maxConffidenceIdx=randsample([1:gdSize2],batchNum2,true,gather(gconffidence2));
        poiPos=gdata2(maxConffidenceIdx, :);
        poiSigma=gsigma2(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata2), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum2)));
        nbPos1=gdata2(nbIdx, :);
        nbSigma1=gsigma2(nbIdx);
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum2)));
        nbPos2=gdata2(nbIdx, :);
        nbSigma2=gsigma2(nbIdx);
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum2)));
        nbPos3=gdata2(nbIdx, :);
        nbSigma3=gsigma2(nbIdx);
        
        w1=repmat(rand(batchNum2, 1), 1, size(gdata2, 2));
        w2=repmat(rand(batchNum2, 1), 1, size(gdata2, 2));
        w3=repmat(rand(batchNum2, 1), 1, size(gdata2, 2));
        
        weight=[];
        for i=1:batchNum2
            wt=[0, w1(i, 1), w2(i, 2), w3(i, 3), 1];
            tmpwt1=wt; tmpwt1(1)=[];
            tmpwt2=wt; tmpwt2(end)=[];
            wt=tmpwt1-tmpwt2;
            weight(i, :)=wt;
        end

        newDataPos=repmat(weight(:, 1), 1, size(gdata2, 2)).*poiPos+...
                   repmat(weight(:, 2), 1, size(gdata2, 2)).*nbPos1+...
                   repmat(weight(:, 3), 1, size(gdata2, 2)).*nbPos2+...
                   repmat(weight(:, 4), 1, size(gdata2, 2)).*nbPos3;
        w1=w1(:, 1)';
        w2=w2(:, 1)';
        
        w1=weight(:, 1)';
        w2=weight(:, 2)';
        w3=weight(:, 3)';
        w4=weight(:, 4)';
        
        newSigma=w1.*poiSigma+w2.*nbSigma1+w3.*nbSigma2+w4.*nbSigma3;
        gsigma2=[gsigma2, newSigma];
        gdata2=[gdata2; newDataPos];
    end
end

%% Simplex, can choose number of vertex in a simplex. No expansion.
function [gdata1, gdata2, gsigma1, gsigma2]=GetNewDataByInterpolationRandomSimplex2(gdata1, gdata2, gsigma1, gsigma2, gconffidence, batchNum, w)
    
    vNum=min(4, size(gdata2, 2)+1);%%%%%%%%%%%%%%%%%%%
    k=min(10, size(gdata2, 1)-1);%%%%%%%%%%%%%%%%%%%
    
    gdSize1=size(gdata1, 1);
    gdSize2=size(gdata2, 1);    
    gconffidence=(gconffidence-min(gconffidence))/(max(gconffidence)-min(gconffidence));
    
    if sum(gconffidence)==0
        gconffidence=zeros(size(gconffidence))+1;
    end
    
    classWeight=[1-w, w];
    class=randsample([1:2],batchNum,true,gather(classWeight));
    
    batchNum1=sum(class==1); %%%%%%%%%%%%%%%%%%%%%%%%
%     batchNum1=ceil(batchNum*0.5);
    if batchNum1~=0
        gconffidence1=gconffidence(1:gdSize1);
        gconffidence1=(gconffidence1-min(gconffidence1))/(max(gconffidence1)-min(gconffidence1));
        gconffidence1=gconffidence1.^1;
        [tmp, idx]=sort(gconffidence1);
        i=uint16(gdSize1*1/gdSize1);
        threshold=tmp(i);
        gconffidence1(gconffidence1<threshold)=0;
        
        if sum(gconffidence1)==0
            gconffidence1=ones(size(gconffidence1));
        end
        
        if numel(find(isnan(gconffidence1)))~=0 || numel(find(isinf(gconffidence1)))~=0
            gconffidence1=ones(size(gconffidence1));
        end
        
        maxConffidenceIdx=randsample([1:gdSize1],batchNum1,true,gather(gconffidence1));
        poiPos=gdata1(maxConffidenceIdx, :);
        poiSigma=gsigma1(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata1), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum1)));
        nbPos=gdata1(nbIdx, :);
        nbSigma=gsigma1(nbIdx);
        w=rand(batchNum1, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata1, 2))+nbPos.*(1-repmat(w, 1, size(gdata1, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma1=[gsigma1, newSigma];
        gdata1=[gdata1; newDataPos];
    end
    
    batchNum2=sum(class==2);
    if batchNum2~=0
        gconffidence2=gconffidence(gdSize1+1:end);
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        gconffidence2=gconffidence2.^1;%%%%%%%%%%%%%%%%%%%%%%%% Normally apply 1.
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        [tmp, idx]=sort(gconffidence2);
        i=uint16(gdSize2*1/gdSize2);
        threshold=tmp(i);
        gconffidence2(gconffidence2<threshold)=0;
        
        if sum(gconffidence2)==0
            gconffidence2=ones(size(gconffidence2));
        end
        
        if numel(find(isnan(gconffidence2)))~=0 || numel(find(isinf(gconffidence2)))~=0
            gconffidence2=ones(size(gconffidence2));
        end
        
        maxConffidenceIdx=randsample([1:gdSize2],batchNum2,true,gather(gconffidence2));
        poiPos=gdata2(maxConffidenceIdx, :);
        poiSigma=gsigma2(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata2), gather(poiPos), 'k', k+1);
        
        nbPos=[];
        nbSigma=[];
        weight=[];
        newDataPos=zeros(batchNum2, size(gdata2, 2));
        newSigma=zeros(batchNum2, 1);
        parfor i=1:size(knnIdx, 1)
            nbs=knnIdx(i, :);
%             rperm = [1, randperm(vNum-1)+1];
            rperm=[1, randi([2, size(nbs, 2)], [1, vNum-1])];
            nbCurPos = gdata2(nbs(rperm), :);
            nbCurSigma = gsigma2(nbs(rperm));
            w = [0, rand(1, vNum-1), 1];
            w = sort(w);
            tmpw1 = w; tmpw1(1)=[];
            tmpw2 = w; tmpw2(end)=[];
            w = tmpw1 - tmpw2;
            wmat = repmat(w', 1, size(nbCurPos, 2));
            newCurDataPos=sum(nbCurPos.*wmat);
            newCurSigma=sum(nbCurSigma.*w);
            newDataPos(i, :)=newCurDataPos;
            newSigma(i)= newCurSigma;
        end
        
        gsigma2=[gsigma2, newSigma'];
        gdata2=[gdata2; newDataPos];
    end
end

function [gdata1, gdata2, gsigma1, gsigma2]=GetNewDataByInterpolationRandomSimplex3(gdata1, gdata2, gsigma1, gsigma2, gconffidence, batchNum, w)
    
    vNum=min(2, size(gdata2, 2)+1);%%%%%%%%%%%%%%%%%%%
    k=min(5, size(gdata2, 1)-1);%%%%%%%%%%%%%%%%%%%
    expansionRate=1.0;%%%%%%%%%%%%%%%%%%%
    
    gdSize1=size(gdata1, 1);
    gdSize2=size(gdata2, 1);    
    gconffidence=(gconffidence-min(gconffidence))/(max(gconffidence)-min(gconffidence));
    
    if sum(gconffidence)==0
        gconffidence=zeros(size(gconffidence))+1;
    end
    
    classWeight=[1-w, w];
    class=randsample([1:2],batchNum,true,gather(classWeight));
    
    batchNum1=sum(class==1); %%%%%%%%%%%%%%%%%%%%%%%%
%     batchNum1=ceil(batchNum*0.5);
    if batchNum1~=0
        gconffidence1=gconffidence(1:gdSize1);
        gconffidence1=(gconffidence1-min(gconffidence1))/(max(gconffidence1)-min(gconffidence1));
        gconffidence1=gconffidence1.^1;
        [tmp, idx]=sort(gconffidence1);
        i=uint16(gdSize1*1/gdSize1);
        threshold=tmp(i);
        gconffidence1(gconffidence1<threshold)=0;
        
        if sum(gconffidence1)==0
            gconffidence1=ones(size(gconffidence1));
        end
        
        if numel(find(isnan(gconffidence1)))~=0 || numel(find(isinf(gconffidence1)))~=0
            gconffidence1=ones(size(gconffidence1));
        end
        
        maxConffidenceIdx=randsample([1:gdSize1],batchNum1,true,gather(gconffidence1));
        poiPos=gdata1(maxConffidenceIdx, :);
        poiSigma=gsigma1(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata1), gather(poiPos), 'k', k+1);
        knnIdx(:, 1)=[];
        nbIdx=diag(knnIdx(:, randi([1, k], 1, batchNum1)));
        nbPos=gdata1(nbIdx, :);
        nbSigma=gsigma1(nbIdx);
        w=rand(batchNum1, 1);
        newDataPos=poiPos.*repmat(w, 1, size(gdata1, 2))+nbPos.*(1-repmat(w, 1, size(gdata1, 2)));
        newSigma=poiSigma.*w'+nbSigma.*(1-w');
        gsigma1=[gsigma1, newSigma];
        gdata1=[gdata1; newDataPos];
    end
    
    batchNum2=sum(class==2);
    if batchNum2~=0
        gconffidence2=gconffidence(gdSize1+1:end);
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        gconffidence2=gconffidence2.^1;%%%%%%%%%%%%%%%%%%%%%%%% Normally apply 1.
        gconffidence2=(gconffidence2-min(gconffidence2))/(max(gconffidence2)-min(gconffidence2));
        [tmp, idx]=sort(gconffidence2);
        i=uint16(gdSize2*1/gdSize2);
        threshold=tmp(i);
        gconffidence2(gconffidence2<threshold)=0;
        
        if sum(gconffidence2)==0
            gconffidence2=ones(size(gconffidence2));
        end
        
        if numel(find(isnan(gconffidence2)))~=0 || numel(find(isinf(gconffidence2)))~=0
            gconffidence2=ones(size(gconffidence2));
        end
        
        maxConffidenceIdx=randsample([1:gdSize2],batchNum2,true,gather(gconffidence2));
        poiPos=gdata2(maxConffidenceIdx, :);
        poiSigma=gsigma2(maxConffidenceIdx);
        [knnIdx, d]=knnsearch(gather(gdata2), gather(poiPos), 'k', k+1);
        
        nbPos=[];
        nbSigma=[];
        weight=[];
        newDataPos=zeros(batchNum2, size(gdata2, 2));
        newSigma=zeros(batchNum2, 1);
        parfor i=1:size(knnIdx, 1)
            nbs=knnIdx(i, :);
%             rperm = [1, randperm(vNum-1)+1];
            rperm=[1, randi([2, size(nbs, 2)], [1, vNum-1])];
            nbCurPos = gdata2(nbs(rperm), :);
            nbCurSigma = gsigma2(nbs(rperm));
            w = [0, rand(1, vNum-1), 1];
            w = sort(w);
            tmpw1 = w; tmpw1(1)=[];
            tmpw2 = w; tmpw2(end)=[];
            w = tmpw1 - tmpw2;
            meanPos = mean(nbCurPos);
            meanPosMat = repmat(meanPos, size(nbCurPos, 1), 1);
            nbCurPos = nbCurPos-meanPosMat;
            wmat = repmat(w', 1, size(nbCurPos, 2));
            newCurDataPos=sum(nbCurPos.*wmat)*expansionRate + meanPos;
            newCurSigma=sum(nbCurSigma.*w);
            newDataPos(i, :)=newCurDataPos;
            newSigma(i)= newCurSigma;
        end
        
        gsigma2=[gsigma2, newSigma'];
        gdata2=[gdata2; newDataPos];
    end
end

function [conffidence]=GetConffidence(pdfToSelf, pdfToAnother, selfSize, otherSize)
     conffidence=log(1+(pdfToSelf./(pdfToAnother))*(1.0*selfSize/otherSize));
%     conffidence=(1.0*pdfToSelf./pdfToAnother)*(1.0*selfSize./otherSize);
end

function [PdfOfPts, Density]=GetPdfOfPoints(srcSigmas, srcPts, tarPts)
% srcSigmas: 1xM
% srcPts: MxF, M observations
% tarPts: NxF, N observations
    lambda=1e-6;
    srcSigmas(find(srcSigmas==0))=lambda;
    srcC=1./sqrt(2*pi.*srcSigmas.^2);
    srcCMat=repmat(srcC, size(tarPts, 1), 1);
    srcSigmaMat=repmat(srcSigmas, size(tarPts, 1), 1);
    srcdSize=size(srcPts, 1);
    GPUexist = ~isempty(which('ginfo'));
    if GPUexist~=0
        distMat=sqrt(max(arrayfun(@plus, sum(tarPts.^2,2), sum(srcPts.^2,2)')-2*(tarPts*srcPts'), 0));
    else
        distMat=sqrt(max(bsxfun(@plus, sum(tarPts.^2,2), sum(srcPts.^2,2)')-2*(tarPts*srcPts'), 0));
    end
    Density= srcCMat .* exp(-distMat.^2./(2.*srcSigmaMat.^2));
    Density(find(Density==0))=lambda;
    PdfOfPts=sum(Density, 2)./srcdSize;
end
