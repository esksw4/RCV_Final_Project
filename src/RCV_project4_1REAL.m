%run('/Users/e.kim4/Downloads/vlfeat-0.9.20/toolbox/vl_setup')
%vl_version verbose
%% 1: Bag Of Words
clear;
close all;
clc;

imgDir = '/Users/e.kim4/Documents/MATLAB/matconvnet-1.0-beta23/class';
imds = imageDatastore(imgDir, 'IncludeSubfolders', true,...
       'LabelSource', 'foldernames');
%split into testing vs train images: split is 10,10
%%Project4 -> Project5: added so it can take diff num of test and training
%%images
numTestImgClass = 10; 
[testImages,trainImages] = splitEachLabel(imds,10);
%label the trainImages 1~10
trainlabels = grp2idx(trainImages.Labels);
sizelabels = size(trainlabels);
concatDesTrain = [];

for i = 1:sizelabels(1)
    train{i} = imread(trainImages.Files{i});
    test{i} = imread(testImages.Files{i});
    %read each train images & test images
    if size(train{i},3) ~= 3
        train{i} = imresize(train{i},[300,300]);
    elseif size(train{i},3) ~= 3
        test{i} = imresize(test{i},[300,300]);
    else
        train{i} = rgb2gray(imresize(train{i},[300,300]));
        test{i} = rgb2gray(imresize(test{i},[300,300]));
    end

    %format them as single (just to run it on vl_sift)
    %singleTrain{i} = single(train{i});
    %singleTest{i} = single(test{i});
    
    %For a subset of all the interstpoints in training image, clustering
    %the descriptors using k-means clustering:
    %output as k-visual words with each word has an associated 128x1
    %centroids 
    %%%USING VL_SIFT: find interest points and descriptor for testing and
    %%%train images
    %[interestTrain{i}, desTrain{i}] = vl_sift(singleTrain{i});
    %[interestTest{i}, desTest{i}] = vl_sift(singleTest{i});
    
    %USING MATLAB BUILT IN FUNCTION
    interestTrain{i} = detectSURFFeatures(train{i});
    %getOnly 150 strongest interestpoint to compute
    interestTrain{i} = interestTrain{i}.selectStrongest(100);
    [desTrain{i}, validPTrain{i}] = extractFeatures(train{i},...
                                    interestTrain{i},'SURFSize',128);
    
    interestTest{i} = detectSURFFeatures(test{i});
    interestTest{i} = interestTest{i}.selectStrongest(100);
    [desTest{i}, validPTest{i}] = extractFeatures(test{i},...
                                  interestTest{i},'SURFSize',128);
    
    %put all the descriptors in one matrix for testing and train images
    concatDesTrain = [concatDesTrain desTrain{i}'];
    %concatDesTest = [concatDesTest desTest{i}];    
end;
%% For K=300 clustering
K = 300;
%so in here, assignments = closest centroids for each of Descriptor.
[centroid, assignTrain] = vl_kmeans(double(concatDesTrain), K); 

%seperate each assigned descriptors into each classes.
for i = 1:sizelabels(1)
   [closestToCentroidsTrain{i}, distanceTrain{i}] = knnsearch(centroid',...
                                                    desTrain{i});
end

for i = 1:sizelabels(1)
   [closestToCentroidsTest{i}, distanceTest{i}] = knnsearch(centroid',...
                                                  desTest{i});
end

%% project4 -> project5 : instead of whole big for loop, I shortened it.
%for i = 1:sizelabels(1)
%    featureVectorTrain{i} = zeros(K,1);
%    featureVectorTest{i} = zeros(K,1);
%    %can be either size of Test or Train
%    sizeDescript = size(closestToCentroidsTest{i});
%    for j=1:sizeDescript(1)
%        featureVectorTrain{i}(closestToCentroidsTrain{i}(j,1),1) =
%        featureVectorTrain{i}(closestToCentroidsTrain{i}(j,1),1) + 1;
%        featureVectorTest{i}(closestToCentroidsTest{i}(j,1),1) =
%        featureVectorTest{i}(closestToCentroidsTest{i}(j,1),1) + 1;
%    end
%    % concatFVTrain's each row contains histogram of eachImages
%    %h = hist(jhbvjdfh, 300);
%    concatFVTrain(:,i) = [featureVectorTrain{i}];
%    concatFVTest(:,i) = [featureVectorTest{i}];
%end
for i = 1:sizelabels(1)
    % each image is represented with 300x1 and there are 50 images
    featureVectorTrain(:,i) = hist(closestToCentroidsTrain{i},K);
    featureVectorTest(:,i) = hist(closestToCentroidsTest{i},K);
end

hist(closestToCentroidsTrain{1},K)


%% Project4 -> project5 : TrainHistogram vs TestHistogram
%%TrainHistogram vs TestHistogram
%compared a set of concatTrainHistogram with TestingHistogram(1). 
%[concatFVTestLabel, concatFVTestDistance] = knnsearch(concatFVTest,concatFVTrain);

%%%%%%%%FOR project4, I did not compute whole train histogram with test
%%%%%%%%histogram. I only did for one histogram vs one train histogram.
%%%%%%%%i=test image number
%%%%%%%%each image has label 1x1 and there are 50 images
%concatFVTestLabel = zeros(sizelabels(1));
%concatFVTestDistance = zeros(sizelabels(1));
for i = 1:sizelabels(1)
   [concatFVTestLabel(i,1), concatFVTestDistance(i,1)] = knnsearch(...
                              featureVectorTrain',featureVectorTest(:,i)');
end

%% Project4 -> project5:classify the concatFVTestLabel that has been
%%labeled to train images,
%%as the class name.
n = 1; %keep track of the ranges.
%%keep track on class number
    for i = 1:size(concatFVTestLabel,1)
        if concatFVTestLabel(i) >= n &&...
           concatFVTestLabel(i) < n+numTestImgClass
                concatFVTestLabel(i) = 1;
        elseif concatFVTestLabel(i) >= n+numTestImgClass...
               && concatFVTestLabel(i) < n+(2*numTestImgClass)
                    concatFVTestLabel(i) = 2;
        elseif concatFVTestLabel(i) >= n+(2*numTestImgClass) ...
               && concatFVTestLabel(i) < n+(3*numTestImgClass)
                    concatFVTestLabel(i) = 3;
        elseif concatFVTestLabel(i) >= n+(3*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(4*numTestImgClass)
                    concatFVTestLabel(i) = 4;
        elseif concatFVTestLabel(i) >= n+(4*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(5*numTestImgClass)
                    concatFVTestLabel(i) = 5;
        elseif concatFVTestLabel(i) >= n+(5*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(6*numTestImgClass)
                    concatFVTestLabel(i) = 6;
        elseif concatFVTestLabel(i) >= n+(6*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(7*numTestImgClass)
                    concatFVTestLabel(i) = 7;
        elseif concatFVTestLabel(i) >= n+(7*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(8*numTestImgClass)
                    concatFVTestLabel(i) = 8;
        elseif concatFVTestLabel(i) >= n+(8*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(9*numTestImgClass)
                    concatFVTestLabel(i) = 9;
        else
            concatFVTestLabel(i) = 10;
        end
    end
%% project4 -> project5: commented it
%% show similar histograms  
%%show different histogram figure(2)
figure(1) 
i = 0;
            subplot(3,2,1);
            hist(closestToCentroidsTrain{1+i},K)
            %featureVectorTest{trackTrain} = dummyTrain.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain(:,1+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(1+i),trainImages.Labels(1+i))); hold off;
          
            i = 4;
            subplot(3,2,2);
            hist(closestToCentroidsTrain{1+i},K)
            %featureVectorTest{trackTest} = dummyTest.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain(:,1+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(1+i),trainImages.Labels(1+i))); hold off;
            
            subplot(3,2,3);
            hist(closestToCentroidsTrain{80+i},K)
            %featureVectorTest{trackTrain} = dummyTrain.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain(:,80+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(80+i),trainImages.Labels(80+i))); hold off;
            
            i = 2;
            subplot(3,2,4);
            hist(closestToCentroidsTrain{80+i},K)
            %featureVectorTest{trackTest} = dummyTest.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain(:,80+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(80+i),trainImages.Labels(80+i))); hold off;
            
            subplot(3,2,5);
            hist(closestToCentroidsTrain{40+i},K)
            %featureVectorTest{trackTrain} = dummyTrain.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain(:,40+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(40+i),trainImages.Labels(40+i))); hold off;
            
            i = 7;
            subplot(3,2,6);
            hist(closestToCentroidsTrain{40+i},K)
            %featureVectorTest{trackTest} = dummyTest.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain(:,40+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(40+i),trainImages.Labels(40+i))); hold off;


%%
% %%ERROR CHECKING
% errorclass = zeros(10,1);
% for i = 1:size(concatFVTestLabel,1)
%     if i>=1 && i <6 && concatFVTestLabel(i) ~= 1
%         errorclass(1) = errorclass(1) + 1/5;
%     elseif i >= 6 && i < 11 && concatFVTestLabel(i) ~= 2
%         errorclass(2) = errorclass(2) + 1/5;
%     elseif i >= 11 && i < 16 && concatFVTestLabel(i) ~= 3
%         errorclass(3) = errorclass(3) + 1/5;
%     elseif i >= 16 && i < 21 && concatFVTestLabel(i) ~= 4
%         errorclass(4) = errorclass(4) + 1/5;    
%     elseif i >= 21 && i < 26 && concatFVTestLabel(i) ~= 5
%         errorclass(5) = errorclass(5) + 1/5;
%     elseif i >= 26 && i < 31 && concatFVTestLabel(i) ~= 6
%         errorclass(6) = errorclass(6) + 1/5;
%     elseif i >= 31 && i < 36 && concatFVTestLabel(i) ~= 7
%         errorclass(7) = errorclass(7) + 1/5;
%     elseif i >= 36 && i < 41 && concatFVTestLabel(i) ~= 8
%         errorclass(8) = errorclass(8) + 1/5;
%     elseif i >= 41 && i < 46 && concatFVTestLabel(i) ~= 9
%         errorclass(9) = errorclass(9) + 1/5;
%     elseif i >= 46 && i < 51 && concatFVTestLabel(i) ~= 10
%         errorclass(10) = errorclass(10) + 1/5;
%     end
% end
%%
%%confusion matrix
stats = confusionmatStats(trainlabels,concatFVTestLabel);
stats.specificity
%% project 4 -> project5 :commented it
%plotting the interest points
% %showing the class butterfly(3),carplate(4),watch(9)
% colors = distinguishable_colors(50);
% figure(11)
% for i = 1:4
%     subplot(2,2,i)
%     imshow(train{10+i})
%     hold on;
%     plot(interestTrain{10+i});
%     title(sprintf('trainImage: butterfly%d with interest points',i))
%     hold off;
% end
% 
% figure(12)
% for i = 1:4
%     subplot(2,2,i)
%     imshow(train{15+i})
%     hold on;
%     plot(interestTrain{15+i});
%     title(sprintf('trainImage: carplate%d with interest points',i))
%     hold off;
% end
% 
figure(13)
i = 1;
%for i = 1:4
    %subplot(2,2,i)
    imshow(train{40+i})
    hold on;
    plot(interestTrain{40+i});
    hold on;
    plot(featureVectorTrain(:,1),centroid(1,:),'r*');
    title(sprintf('trainImage: carplate%d with interest points',i))
    hold off;
%end

%plot(interestTrain{1}.Location(11,1),interestTrain{1}.Location(11,2),'*');
%plot(interestTrain{1}.Location(244,1),interestTrain{1}.Location(258,2)'*');

%% For K=200 clustering
K200 = 200;
%so in here, assignments = closest centroids for each of Descriptor.
[centroid200, assignTrain200] = vl_kmeans(double(concatDesTrain), K200);

%seperate each assigned descriptors into each classes.
for i = 1:sizelabels(1)
   [closestToCentroidsTrain200{i}, distanceTrain200{i}] = knnsearch(...
                                                centroid200',desTrain{i});
end

for i = 1:sizelabels(1)
    [closestToCentroidsTest200{i}, distanceTest200{i}] = knnsearch(...
                                                 centroid200',desTest{i});
end

%% project4 -> project5:: create Histogram for Train and Testing
for i = 1:sizelabels(1)
    % each image is represented with 300x1 and there are 50 images
    featureVectorTrain200(:,i) = hist(closestToCentroidsTrain200{i},K);
    featureVectorTest200(:,i) = hist(closestToCentroidsTest200{i},K);
end

%% Project4 -> project5 : TrainHistogram vs TestHistogram
%compared a set of concatTrainHistogram with TestingHistogram(1).
for i = 1:sizelabels(1)
    [concatFVTestLabel200(i,1), concatFVTestDistance200(i,1)] =knnsearch...
                       (featureVectorTrain200',featureVectorTest200(:,i)');
end
%% Project4 -> project5: classify between class 1~10
n = 1; %keep track of the ranges.
%%keep track on class number
    for i = 1:size(concatFVTestLabel200,1)
        if concatFVTestLabel200(i) >= n &&...
           concatFVTestLabel200(i) < n+numTestImgClass
                concatFVTestLabel200(i) = 1;
        elseif concatFVTestLabel200(i) >= n+numTestImgClass...
               && concatFVTestLabel200(i) < n+(2*numTestImgClass)
                    concatFVTestLabel200(i) = 2;
        elseif concatFVTestLabel200(i) >= n+(2*numTestImgClass) ...
               && concatFVTestLabel200(i) < n+(3*numTestImgClass)
                    concatFVTestLabel200(i) = 3;
        elseif concatFVTestLabel200(i) >= n+(3*numTestImgClass)... 
               && concatFVTestLabel200(i) < n+(4*numTestImgClass)
                    concatFVTestLabel200(i) = 4;
        elseif concatFVTestLabel200(i) >= n+(4*numTestImgClass)... 
               && concatFVTestLabel200(i) < n+(5*numTestImgClass)
                    concatFVTestLabel200(i) = 5;
        elseif concatFVTestLabel200(i) >= n+(5*numTestImgClass)... 
               && concatFVTestLabel200(i) < n+(6*numTestImgClass)
                    concatFVTestLabel200(i) = 6;
        elseif concatFVTestLabel200(i) >= n+(6*numTestImgClass)... 
               && concatFVTestLabel200(i) < n+(7*numTestImgClass)
                    concatFVTestLabel200(i) = 7;
        elseif concatFVTestLabel200(i) >= n+(7*numTestImgClass)... 
               && concatFVTestLabel200(i) < n+(8*numTestImgClass)
                    concatFVTestLabel200(i) = 8;
        elseif concatFVTestLabel200(i) >= n+(8*numTestImgClass)... 
               && concatFVTestLabel200(i) < n+(9*numTestImgClass)
                    concatFVTestLabel200(i) = 9;
        else
            concatFVTestLabel200(i) = 10;
        end
    end
%%
%confusion matrix K=200
clc
stats200 = confusionmatStats(trainlabels,concatFVTestLabel200);

%%
figure(1) 
            subplot(3,2,1);
            hist(closestToCentroidsTrain200{1+i},K)
            %featureVectorTest{trackTrain} = dummyTrain.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain200(:,1+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(1+i),trainImages.Labels(1+i))); hold off;
          
            i = 4;
            subplot(3,2,2);
            hist(closestToCentroidsTrain200{1+i},K)
            %featureVectorTest{trackTest} = dummyTest.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain200(:,1+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(1+i),trainImages.Labels(1+i))); hold off;
            
            subplot(3,2,3);
            hist(closestToCentroidsTrain200{80+i},K)
            %featureVectorTest{trackTrain} = dummyTrain.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain200(:,80+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(80+i),trainImages.Labels(80+i))); hold off;
            
            i = 2;
            subplot(3,2,4);
            hist(closestToCentroidsTrain200{80+i},K)
            %featureVectorTest{trackTest} = dummyTest.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain200(:,80+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(80+i),trainImages.Labels(80+i))); hold off;
            
            subplot(3,2,5);
            hist(closestToCentroidsTrain200{40+i},K)
            %featureVectorTest{trackTrain} = dummyTrain.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain200(:,40+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(40+i),trainImages.Labels(40+i))); hold off;
            
            i = 7;
            subplot(3,2,6);
            hist(closestToCentroidsTrain200{40+i},K)
            %featureVectorTest{trackTest} = dummyTest.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain200(:,40+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(40+i),trainImages.Labels(40+i))); hold off;

%%

