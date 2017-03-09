%%
%run('/Users/e.kim4/Downloads/vlfeat-0.9.20/toolbox/vl_setup')
%vl_version verbose
%%
%untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta23.tar.gz') ;
%cd matconvnet-1.0-beta23
%urlwrite(...
%  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
%  'imagenet-vgg-f.mat') ;
%urlwrite(...
%  'http://www.vlfeat.org/matconvnet/models/imagenet-matconvnet-vgg-verydeep-16.mat', ...
%  'imagenet-matconvnet-vgg-verydeep-16.mat') ;
%%
%cd matconvnet-1.0-beta23
run matlab/vl_setupnn ;
%%
close all;
clear all;

net1 = load('imagenet-vgg-f.mat');
net2 = load('imagenet-matconvnet-vgg-verydeep-16.mat');
net1 = vl_simplenn_tidy(net1);
net2 = vl_simplenn_tidy(net2);

%%
imgDir = '/Users/e.kim4/Documents/MATLAB/matconvnet-1.0-beta23/class';
imds = imageDatastore(imgDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%size of classes
sizeClasses = size(imds.Labels);
%%
trainLabel=[];
for i = 1:sizeClasses(1)
    %each class contains 5 images; therefore total 50 images.
    %each image read into each cell
    cnntestimage{i} = imresize(single(imread(imds.Files{i})),...
                net1.meta.normalization.imageSize(1:2));
    %image - meanimage
    cnntestimage{i} = cnntestimage{i} - net1.meta.normalization.averageImage;
    %find the trainLabel from .mat
    trainLabel(i,1)= strmatch(imds.Labels(i),net1.meta.classes.description');
    
    %do CNN: each output read into each cell
    cnnOutput1{i} = vl_simplenn(net1,cnntestimage{i});
    cnnOutput2{i} = vl_simplenn(net2,cnntestimage{i});
    %Find the maximum similarity and assign that test image
    cnnscores1{i} = squeeze(gather(cnnOutput1{i}(end).x));
    cnnscores2{i} = squeeze(gather(cnnOutput2{i}(end).x));
    [cnnBestScore1(i,1), cnnScoreLoc1(i,1)] = max(cnnscores1{i});
    [cnnBestScore2(i,1), cnnScoreLoc2(i,1)] = max(cnnscores2{i});

%     %Plotting
    if i == 1 
        figure(1);
    imshow(imread(imds.Files{i}));
    title(sprintf('%s %s (%d),\n%s %s (%d),\n %s %s,\n%s %.3f\n%s %.3f',...
        'CNN classification1:',...
        net1.meta.classes.description{cnnScoreLoc1(i,1)}, cnnScoreLoc1(i,1),...
        'CNN classification2:',...
        net2.meta.classes.description{cnnScoreLoc2(i,1)}, cnnScoreLoc2(i,1),...
        'Test image:', imds.Labels(i),...
        'Score1:',cnnBestScore1(i,1),...
        'Score2:',cnnBestScore2(i,1)));
%     elseif i ==6
%         figure(2);
%         subplotNumber = 1;
%     elseif i==11
%         figure(3);
%         subplotNumber = 1;
    elseif i==16
        figure(4);

    imshow(imread(imds.Files{i}));
    title(sprintf('%s %s (%d),\n%s %s (%d),\n %s %s,\n%s %.3f\n%s %.3f',...
        'CNN classification1:',...
        net1.meta.classes.description{cnnScoreLoc1(i,1)}, cnnScoreLoc1(i,1),...
        'CNN classification2:',...
        net2.meta.classes.description{cnnScoreLoc2(i,1)}, cnnScoreLoc2(i,1),...
        'Test image:', imds.Labels(i),...
        'Score1:',cnnBestScore1(i,1),...
        'Score2:',cnnBestScore2(i,1)));

%     elseif i==21
%         figure(5);
%         subplotNumber = 1;
%     elseif i==26
%         figure(6);
%         subplotNumber = 1;
%     elseif i==31
%         figure(7);
%         subplotNumber = 1;     
%     elseif i==36
%         figure(8);
%         subplotNumber = 1;
    elseif i==41
        figure(9);
    imshow(imread(imds.Files{i}));
    title(sprintf('%s %s (%d),\n%s %s (%d),\n %s %s,\n%s %.3f\n%s %.3f',...
        'CNN classification1:',...
        net1.meta.classes.description{cnnScoreLoc1(i,1)}, cnnScoreLoc1(i,1),...
        'CNN classification2:',...
        net2.meta.classes.description{cnnScoreLoc2(i,1)}, cnnScoreLoc2(i,1),...
        'Test image:', imds.Labels(i),...
        'Score1:',cnnBestScore1(i,1),...
        'Score2:',cnnBestScore2(i,1))); 
%     elseif i== 46
%         figure(10);
%         subplotNumber = 1;
    end
%     subplot(3,1,subplotNumber)
%     imshow(imread(imds.Files{i}));
%     title(sprintf('%s %s (%d),\n%s %s (%d),\n %s %s,\n%s %.3f\n%s %.3f',...
%         'CNN classification1:',...
%         net1.meta.classes.description{cnnScoreLoc1(i,1)}, cnnScoreLoc1(i,1),...
%         'CNN classification2:',...
%         net2.meta.classes.description{cnnScoreLoc2(i,1)}, cnnScoreLoc2(i,1),...
%         'Test image:', imds.Labels(i),...
%         'Score1:',cnnBestScore1(i,1),...
%         'Score2:',cnnBestScore2(i,1)));
%     subplotNumber = subplotNumber +1;   
end

%% classify the wrong label as others.
%%trainLabel = 249/251, 22, 230, 45, 208, 85,615, 110, 131, 237
%%if cnnScoreLoc1 and cnnScoreLoc2 have anything besides trainLabel,
%%classify as 1000

for i=1:size(cnnScoreLoc1,1)
    cnnScoreLoc1_1(i,1) = cnnScoreLoc1(i);
    if cnnScoreLoc1(i) ~= trainLabel(:)
        cnnScoreLoc1_1(i) = 1001;
    end
end

for i=1:size(cnnScoreLoc2,1)
    cnnScoreLoc2_1(i,1) = cnnScoreLoc2(i);
    if cnnScoreLoc2(i) ~= trainLabel(:)
        cnnScoreLoc2_1(i) = 1001;
    end
end
%% confusion matrix
stat1 = confusionmatStats(trainLabel,cnnScoreLoc1_1);
stat2 = confusionmatStats(trainLabel,cnnScoreLoc2_1);



    