function patches = sampleIMAGES(patchsize,numpatches)
% sampleIMAGES
% Returns 10000 patches for training

IMAGES = loadMNISTImages('train-images.idx3-ubyte'); % load images from disk 

% Initialize patches with zeros.
patches = zeros(patchsize*patchsize, numpatches);

%  IMAGES is a 3D array containing 10 images

[XY,N] = size(IMAGES);

for i = 1:numpatches
    
    imIndex = ceil(N*rand);
    
    imageSample = IMAGES(:,imIndex);
    
    patches(:,i) = imageSample(:);
end

% For the autoencoder to work well it is needed to normalize the data
patches = normalizeData(patches);

end


function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
