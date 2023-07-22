%% variables
patternDirectory = fullfile('data', 'test_masks');
patternFilename = 'sampleD2NN_dZspacing=1.7um.tiff';

patternPixelPitch_um = 589 / 1024;
patternZPitch_um = 1.7;

computePixelPitch_um = patternPixelPitch_um / 2;

psfFWHM = [1.2, 1.2, 4.0];

%% setup
defaultProjViewArgs = {
    'DarkMode', true, ...
    'UnitRatio', computePixelPitch_um, ...
    'UnitName', 'um', ...
    'ProjectionFcn', 'sum', ...
    'DoShowAxes', false};

%% load image
patternFilepath = fullfile(patternDirectory, patternFilename);
refPatternRaw = csmu.volread(patternFilepath);

xVecRaw = ((1:size(refPatternRaw, 1)) - 0.5) * patternPixelPitch_um;
yVecRaw = ((1:size(refPatternRaw, 2)) - 0.5) * patternPixelPitch_um;
zVecRaw = ((1:size(refPatternRaw, 3)) - 0.5) * patternZPitch_um;

newSize = floor(size(refPatternRaw) ...
    .* [[1, 1] * patternPixelPitch_um, patternZPitch_um] ...
    / computePixelPitch_um);

xVecNew = ((1:newSize(1)) - 0.5) * computePixelPitch_um;
yVecNew = ((1:newSize(2)) - 0.5) * computePixelPitch_um;
zVecNew = ((1:newSize(3)) - 0.5) * computePixelPitch_um;

xVecNewId = ceil(xVecNew / patternPixelPitch_um);
yVecNewId = ceil(yVecNew / patternPixelPitch_um);
zVecNewId = ceil(zVecNew / patternZPitch_um);

refPattern = refPatternRaw(xVecNewId, yVecNewId, zVecNewId);

csplot.quick.projView(refPattern, ...
   defaultProjViewArgs{:}, 'ScaleBarLength', 50);

%% generate psf
psfSigma = psfFWHM / (2 * sqrt(2 * log(2)));

gaussian3d = @(x, y, z) exp(-1 * (...
    ((x .^ 2) / 2 / (psfSigma(1) ^ 2)) ...
    + ((y .^ 2) / 2 / (psfSigma(2) ^ 2)) ...
    + ((z .^ 2) / 2 / (psfSigma(3) ^ 2))));

volWidth = ceil(psfSigma * 10 / computePixelPitch_um);
% make even valued
volWidth = (round(volWidth / 2) * 2);

[xVec, yVec, zVec] = csmu.zeroCenterVector(volWidth);
xVec = xVec * computePixelPitch_um;
yVec = yVec * computePixelPitch_um;
zVec = zVec * computePixelPitch_um;

[xGrid, yGrid, zGrid] = meshgrid(xVec, yVec, zVec);

psf = gaussian3d(xGrid, yGrid, zGrid);
psf = psf / sum(psf, "all");

csplot.quick.projView(psf, defaultProjViewArgs{:}, ...
    'ScaleBarLength', 3);


%% setup for deconvolution


%% define convolution functions
function pattern = forwardConv(mask, psf, patternSpace, ...
    forwardZSelArray)
    tempOutput = zeros([size(patternSpace, [1, 2]), size(psf, 3)], ...
        'like', patternSpace);    
    pattern = patternSpace;
    for iMaskPlane = 1:size(mask, 3)
        tempOutput(:) = 0;
        maskPage = mask(:, :, iMaskPlane);
        for iPsfPlane = 1:size(psf, 3)
            psfPage = psf(:, :, iPsfPlane);            
            tempOutput(:, :, iPsfPlane) = conv2(maskPage, psfPage, 'same');
        end    
        [patternSpaceZSel, convZSel] = forwardZSelArray{iMaskPlane, :};
        pattern(:, :, patternSpaceZSel) = ...
            pattern(:, :, patternSpaceZSel) ...
            + tempOutput(:, :, convZSel);
    end
end

function mask = backwardConv(pattern, psf, maskSpace, ...
    backwardZSelArray, method)
    tempOutput = convn(pattern, psf, 'same');
    mask = maskSpace;
    for iMaskPlane = 1:size(maskSpace, 3)        
        [patternSpaceZSel, patternSpacePage] = ...
            backwardZSelArray{iMaskPlane, :};
        switch lower(method)
            case 'mean'
                mask(:, :, iMaskPlane) = ...
                    mean(tempOutput(:, :, patternSpaceZSel), 3);
            case 'page'
                mask(:, :, iMaskPlane) = ...
                    tempOutput(:, :, patternSpacePage);
            otherwise
                error('Unexpected value for method recieved, "%s".', ...
                    method);
        end
    end
end