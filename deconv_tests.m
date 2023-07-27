%% variables
patternDirectory = fullfile('data', 'test_masks');
patternFilename = 'sampleD2NN_dZspacing=1.7um.tiff';

patternPixelPitch_um = 589 / 1024;
patternZPitch_um = 1.7;

computePixelPitch_um = patternPixelPitch_um / 2;

% xy-order
psfFWHM = [1.2, 1.2, 4.0];

maskPixelPitch_um = patternPixelPitch_um;
maskZPitch_um = patternZPitch_um;

numIterations = 20;
iterGaussSigma = 0;
outputNum = 003;

patternZPadSize = 3;

%% setup
L = csmu.Logger(mfilename);
L.windowLevel = csmu.LogLevel.DEBUG;

defaultProjViewArgs = {
    'DarkMode', true, ...    
    'UnitName', 'um', ...
    'ProjectionFcn', 'sum', ...
    'DoShowAxes', false};

%% load image
L.info('Loading Desired Pattern File');

patternFilepath = fullfile(patternDirectory, patternFilename);
refPatternRaw = csmu.volread(patternFilepath);
refPatternRaw = im2double(refPatternRaw);
refPatternRaw = padarray(refPatternRaw, [0, 0, ...
    patternZPadSize], 0, 'both');

xVecRaw = ((1:size(refPatternRaw, 1)) - 0.5) * patternPixelPitch_um;
yVecRaw = ((1:size(refPatternRaw, 2)) - 0.5) * patternPixelPitch_um;
zVecRaw = ((1:size(refPatternRaw, 3)) - 0.5) * patternZPitch_um;

newSize = round(size(refPatternRaw) ...
    .* [[1, 1] * patternPixelPitch_um, patternZPitch_um] ...
    / computePixelPitch_um);

% col
patternXVec = ((1:newSize(2)) - 0.5) * computePixelPitch_um;
% row
patternYVec = ((1:newSize(1)) - 0.5) * computePixelPitch_um;
patternZVec = ((1:newSize(3)) - 0.5) * computePixelPitch_um;

xVecNewId = ceil(patternXVec / patternPixelPitch_um);
yVecNewId = ceil(patternYVec / patternPixelPitch_um);
zVecNewId = ceil(patternZVec / patternZPitch_um);

refPattern = refPatternRaw(yVecNewId, xVecNewId, zVecNewId);

csplot.quick.projView(refPattern, ...
   defaultProjViewArgs{:}, ...
   'ScaleBarLength', 50, ...
   'FigureName', 'Target Pattern', ...
   'UnitRatio', computePixelPitch_um);

%% generate psf
L.info('Computing PSF');

psfSigma_um = psfFWHM / (2 * sqrt(2 * log(2)));

gaussian3d = @(x, y, z) exp(-1 * (...
    ((x .^ 2) / 2 / (psfSigma_um(1) ^ 2)) ...
    + ((y .^ 2) / 2 / (psfSigma_um(2) ^ 2)) ...
    + ((z .^ 2) / 2 / (psfSigma_um(3) ^ 2))));

psfSize = ceil(psfSigma_um * 6 / computePixelPitch_um);
% make psfSize even valued
psfSize = (round(psfSize / 2) * 2);

% row-col to x-y
[psfYVec, psfXVec, psfZVec] = csmu.zeroCenterVector(psfSize);
psfXVec = psfXVec * computePixelPitch_um;
psfYVec = psfYVec * computePixelPitch_um;
psfZVec = psfZVec * computePixelPitch_um;

% me sure to switch from x-y to row-col
[yGrid, xGrid, zGrid] = meshgrid(psfYVec, psfXVec, psfZVec);

psf = gaussian3d(xGrid, yGrid, zGrid);
psf = psf / sum(psf, "all");

csplot.quick.projView(psf, defaultProjViewArgs{:}, ...
    'ScaleBarLength', 3, ...
    'FigureName', 'PSF', ...
    'UnitRatio', computePixelPitch_um);


%% setup for deconvolution
L.info('Setting up for deconvolution');
patternSpaceSize = size(refPattern);
patternSpace = zeros(patternSpaceSize);

% FIXME - do zero-centered indexing and use imref's with extents
maskSpaceSize = round((patternSpaceSize) ...
    .* [1, 1, 1] * computePixelPitch_um ...
    ./ [[1, 1] * maskPixelPitch_um, maskZPitch_um]);
maskSpace = zeros(maskSpaceSize);
maskXVec = ((1:maskSpaceSize(2)) - 0.5) * maskPixelPitch_um; % col
maskYVec = ((1:maskSpaceSize(1)) - 0.5) * maskPixelPitch_um; % row
maskZVec = ((1:maskSpaceSize(3)) - 0.5) * maskZPitch_um;

% FIXME - add substack weights
patternSubStackZVec = csmu.zeroCenterVector(...
    ceil(maskZPitch_um / computePixelPitch_um)) * computePixelPitch_um;

maskXPatternIdVec = ceil(maskXVec / computePixelPitch_um);
maskYPatternIdVec = ceil(maskYVec / computePixelPitch_um);
maskZPatternIdVec = ceil(maskZVec / computePixelPitch_um);

patternXMaskIdVec = ceil(patternXVec / maskPixelPitch_um);
patternYMaskIdVec = ceil(patternYVec / maskPixelPitch_um);

computeToPageRatio = maskPixelPitch_um / computePixelPitch_um;

if computeToPageRatio == 1
    maskPageToComputeRes = @(x) x;
    patternPageToMaskRes = @(x) x;
else
    assert(computeToPageRatio > 1);
    maskPageToComputeRes = @(page) ...
        page(patternYMaskIdVec, patternXMaskIdVec);
    if csmu.isint(computeToPageRatio)
        patternPageToMaskRes = @(page) ...
            averagingDownsample(page, computeToPageRatio);
    else
        patternPageToMaskRes = @(page) ...
            blurDownsample(page, maskXPatternIdVec, ...
            maskYPatternIdVec);
    end

end

forwPatternSpaceZSel = cell(maskSpaceSize(3), 1);
forwConvZSel = cell(maskSpaceSize(3), 1);

for iMaskPlane = 1:maskSpaceSize(3)
    patternZSel = ceil(...
        (maskZVec(iMaskPlane) + psfZVec) / computePixelPitch_um);
    valid = (patternZSel >= 1) & (patternZSel <= patternSpaceSize(3));

    forwPatternSpaceZSel{iMaskPlane} = patternZSel(valid);
    
    convZSel = 1:psfSize(3);
    forwConvZSel{iMaskPlane} = convZSel(valid);
end

forwardZSelArray = [forwPatternSpaceZSel, forwConvZSel];

backPatternSpaceZSel = cell(maskSpaceSize(3), 1);
backPatternSpacePage = cell(maskSpaceSize(3), 1);

for iMaskPlane = 1:maskSpaceSize(3)
    patternZSel = ceil(...
        (maskZVec(iMaskPlane) + patternSubStackZVec) ...
        / computePixelPitch_um);
    valid = (patternZSel >= 1) & (patternZSel <= patternSpaceSize(3));
    backPatternSpaceZSel{iMaskPlane} = patternZSel(valid);

    backPatternSpacePage{iMaskPlane} = maskZPatternIdVec(iMaskPlane);
end

backwardZSelArray = [backPatternSpaceZSel, backPatternSpacePage];

maskToPattern = @(mask) forwardConv(...
    mask, ...
    psf, ...
    patternSpace, ...
    forwardZSelArray, ...
    maskPageToComputeRes);

patternToMask = @(pattern) backwardConv(...
    pattern, ...
    psf, ...
    maskSpace, ...
    backwardZSelArray, ...
    patternPageToMaskRes, ...
    'mean');

%% generate correction projections
L.info('Generating Correction Projections')
t = tic();
forwCorrection = maskToPattern(maskSpace + 1);
L.info('maskToPattern took %s.', csmu.durationString(toc(t)));
L.debug('Range of forward correction is [%s]', ...
    num2str(csmu.range(forwCorrection, 'all')))

t = tic();
backCorrection = patternToMask(patternSpace + 1);
L.info('patternToMask took %s.', csmu.durationString(toc(t)));
L.debug('Range of back correction is [%s]', ...
    num2str(csmu.range(backCorrection, 'all')))

t = tic();
testSpace = maskSpace;
center = round(maskSpaceSize / 2);
testSpace(center(1), center(2), center(3)) = 1;
testForwProj = maskToPattern(testSpace);
L.info('maskToPattern took %s.', csmu.durationString(toc(t)));

L.debug('Range of test projetion (pre correction) is [%s]', ...
    num2str(csmu.range(testForwProj, 'all')));
testForwProj = testForwProj ./ forwCorrection;
L.debug('Range of test projection (post correction) is [%s]', ...
    num2str(csmu.range(testForwProj, 'all')))

%% preview correction projections
% csplot.quick.projView(forwCorrection, defaultProjViewArgs{:}, ...
%     'ScaleBarLength', 50, ...
%     'FigureName', 'ONES-Mask->Pattern', ...
%     'UnitRatio', computePixelPitch_um);
% 
% csplot.quick.projView(backCorrection, defaultProjViewArgs{:}, ...
%     'ScaleBarLength', 50, ...
%     'FigureName', 'ONES-Pattern->Mask', ...
%     'UnitRatio', [[1, 1] * maskPixelPitch_um, maskZPitch_um]);

csplot.quick.projView(testForwProj, defaultProjViewArgs{:}, ...
    'ScaleBarLength', 50, ...
    'FigureName', 'Test-Mask->Pattern', ...
    'UnitRatio', computePixelPitch_um);

%% initial guess

% FIXME - add more choices for initial guess
maskGuess = patternToMask(refPattern);

L.debug('Range of initial mask (pre correction) is [%s]', ...
    num2str(csmu.range(maskGuess, 'all')));

maskGuess = maskGuess ./ backCorrection;

L.debug('Range of initial mask (post correction) is [%s]', ...
    num2str(csmu.range(maskGuess, 'all')));

csplot.quick.projView(maskGuess, defaultProjViewArgs{:}, ...
    'ScaleBarLength', 50, ...
    'FigureName', 'Initial Mask (Raw)', ...
    'UnitRatio', [[1, 1] * maskPixelPitch_um, maskZPitch_um]);

maskGuess = csmu.bound(maskGuess, 0, 1);

csplot.quick.projView(double(maskGuess > 0.5), defaultProjViewArgs{:}, ...
    'ScaleBarLength', 50, ...
    'FigureName', 'Initial Mask (Binary)', ...
    'UnitRatio', [[1, 1] * maskPixelPitch_um, maskZPitch_um]);

% csplot.quick.projView(double(maskGuess > 0.5) - refPatternRaw, ...
%     defaultProjViewArgs{:}, ...
%     'ScaleBarLength', 50, ...
%     'FigureName', 'Diff From Ref Mask', ...
%     'Colormap', 'Cool', ...
%     'ColorLimits', [-1, 1], ...
%     'UnitRatio', [[1, 1] * maskPixelPitch_um, maskZPitch_um]);
% 
% figure('Name', 'Mask Guess');
% csmu.imshow3d(maskGuess - refPatternRaw, [-1, 1], 'cool');


%% deconvolution
L.info('Beginning Deconvolution.');

f = figure(85);
f.Name = 'Error Tracking';
errorNorm = sum(refPattern, 'all');
duplicatedMask = maskSpace;
for iPage = 1:maskSpaceSize(3)
    duplicatedMask(:, :, iPage) = ...
        patternPageToMaskRes(...
        mean(refPattern(:, :, backPatternSpaceZSel{iPage}), 3));
end
errRef = maskToPattern(duplicatedMask) ./ forwCorrection;
errRef = sum(abs(refPattern - errRef), 'all') / errorNorm;
plot([0, 20], [errRef, errRef], '-.k', 'LineWidth', 2, ...
    'DisplayName', 'Reference (2X compute, patterns as-is)');
hold('on');
p = plot(0, 0, '.:', 'MarkerSize', 20, 'LineWidth', 2, 'DisplayName', ...
    '2X compute, z-pad-2, 0.5 init, momentum [0.3,0.05,0.01] update weight, NO-Gaussian');
grid('on');
hold('on');

title('Error Tracking');
xlabel('Iteration Index');
ylabel('Error Metric (a.u.)');
p.XData = [];
p.YData = [];

legend();

maskGuess(:) = 0.5;

updateHistory = repmat(maskSpace, 1, 1, 1, 3);
weights = reshape(4 .^ (-1 * (1:3)), 1, 1, 1, []);
weights(:) = [0.3, 0.05, 0.01];
validUpdates = reshape([false, false, false], 1, 1, 1, []);

bestErr = [];
bestMask = [];
bestIter = [];
increaseCount = 0;

lastErrorMetric = [];

for iIteration = 0:numIterations
    t = tic();
    if iIteration > 0   
        % `forward` is computed at the end of the prior iteration
        errorPattern = refPattern ./ forward;
        backward = patternToMask(errorPattern) ./ backCorrection;
        maskGuessRaw = backward .* maskGuess;
        % maskGuess =  double(maskGuessRaw > 0.5);
        % maskGuess = csmu.bound(maskGuessRaw, 0, 1);
        maskGuessRaw = csmu.bound(maskGuessRaw, 0, 1);

        updateDelta = maskGuessRaw - maskGuess;
        updateHistory(:, :, :, 2:3) = updateHistory(:, :, :, 1:2);
        updateHistory(:, :, :, 1) = updateDelta;
        validUpdates(:) = cat(4, true, validUpdates(1:2));
        validWeights = weights(validUpdates);
        % validWeights = validWeights / sum(validWeights, 'all');
        netUpdateDelta = sum(updateHistory(:, :, :, validUpdates) ...
            .* validWeights, 4);  
        maskGuess = maskGuess + netUpdateDelta;
        % maskGuess = csmu.bound(maskGuess, 0, 1);
        if ~(iterGaussSigma == 0)
            maskGuess = imgaussfilt(maskGuess, iterGaussSigma, ... 
                'FilterDomain', 'spatial');
        end
        % maskGuess = double(maskGuess > 0.5);

        L.debug('Range of mask guess is [%s]', ...
            num2str(csmu.range(maskGuess, 'all')));
    end

    forward = maskToPattern(maskGuess) ./ forwCorrection; 
    
    L.info('Iteration %d took %s.', iIteration, ...
        csmu.durationString(toc(t)));

    maskGuessBin = double(maskGuess > 0.5);

    % if mod(iIteration, 4) == 1
    %     figure('Name', sprintf('Mask at Iteration %d', iIteration));
    %     csmu.imshow3d(maskGuessBin, [0, 1]);
    % end
    
    forwardTest = maskToPattern(maskGuessBin) ./ forwCorrection; 
    errorMetric = sum(abs(refPattern - forwardTest), 'all') / errorNorm;
    p.XData = [p.XData, iIteration];
    p.YData = [p.YData, errorMetric];
    drawnow();

    if isempty(bestErr)
        bestErr = errorMetric;
        bestMask = maskGuess;
        bestIter = iIteration;
    end

    if errorMetric < bestErr
        bestErr = errorMetric;
        bestMask = maskGuess;
        bestIter = iIteration;
    end

    if iIteration >= 1
        if errorMetric > lastErrorMetric
            L.info('Error Metric Increased!');
            increaseCount = increaseCount + 1;
        else
            increaseCount = 0;
        end
    end
    lastErrorMetric = errorMetric;

    if increaseCount > 1
        L.info('Deconvolution has halted converging, Exiting.');
        break
    end
end

%% choose best threshold
L.info('Selecting Threshold');

f = figure();
f.Name = 'Threshold Select';

pT = plot(0, NaN, 'k.-', 'MarkerSize', 20, 'LineWidth', 2, ...
    'DisplayName', 'tests');
hold('on');
p2 = plot(0, NaN, 'ro', 'MarkerSize', 5, 'LineWidth', 2, ...
    'MarkerFaceColor', 'none', ...
    'DisplayName', 'best');
legend;
grid('on');

title(sprintf('Threshold Selection: %s', p.DisplayName));
xlabel('Threshold Value');
ylabel('Error Metric (a.u.)');
pT.XData = [];
pT.YData = [];
bestErr = [];
bestThresh = [];
thresholds = linspace(0.5 - 0.01, 0.5, 9);
numThresh = length(thresholds);
for iThresh = 1:numThresh
    maskGuessBin = double(bestMask > thresholds(iThresh));       
    forwardTest = maskToPattern(maskGuessBin) ./ forwCorrection;
    errorMetric = sum(abs(refPattern - forwardTest), 'all') / errorNorm;

    if isempty(bestErr)
        bestErr = errorMetric;
        bestThresh = thresholds(iThresh);
    else
        if errorMetric < bestErr
            bestErr = errorMetric;
            bestThresh = thresholds(iThresh);
        end
    end

    pT.XData = [pT.XData, thresholds(iThresh)];
    pT.YData = [pT.YData, errorMetric];
    p2.XData = bestThresh;
    p2.YData = bestErr;
    drawnow();
end

%% plot results
computedMask = double(bestMask > bestThresh);


figure('Name', 'Computed Mask');
csmu.imshow3d(computedMask, [0, 1]);

csplot.quick.projView(computedMask, defaultProjViewArgs{:}, ...
    'ScaleBarLength', 50, ...
    'FigureName', 'Computed Mask', ...
    'UnitRatio', [[1, 1] * maskPixelPitch_um, maskZPitch_um]);

csplot.quick.projView(computedMask - duplicatedMask, ...
    defaultProjViewArgs{:}, ...
    'ScaleBarLength', 50, ...
    'FigureName', 'Diff From Ref Mask', ...
    'Colormap', 'Cool', ...
    'ColorLimits', [-1, 1], ...
    'UnitRatio', [[1, 1] * maskPixelPitch_um, maskZPitch_um]);

simPattern = maskToPattern(computedMask) ./ forwCorrection;

csplot.quick.projView(simPattern - refPattern, ...
    defaultProjViewArgs{:}, ...
    'ScaleBarLength', 50, ...
    'FigureName', 'Sim. Diff From Ref Pattern', ...
    'Colormap', 'Cool', ...
    'ColorLimits', [-2, 2], ...
    'UnitRatio', computePixelPitch_um);

csplot.quick.projView(abs(simPattern - refPattern), ...
    defaultProjViewArgs{:}, ...
    'ScaleBarLength', 50, ...
    'FigureName', 'Sim. abs(Diff) From Ref Pattern', ...
    'UnitRatio', computePixelPitch_um);

figure('Name', 'Sim. abs(Diff) From Ref Pattern, paged');
csmu.imshow3d(abs(simPattern - refPattern));

figure('Name', 'Ref Pattern');
csmu.imshow3d(refPattern);

figure('Name', 'Sim. Pattern from Mask');
csmu.imshow3d(simPattern);

baselineSim = maskToPattern(duplicatedMask) ./ forwCorrection;
figure('Name', 'Sim. Pattern from NO-DECONV Mask');
csmu.imshow3d(baselineSim);

csmu.volume2tif(computedMask, ...
    sprintf('analysis\\%03d-computed-deconv-mask.tif', outputNum), 'Class', 'uint8');
csmu.volume2tif(refPattern,...
    sprintf('analysis\\%03d-ref-pattern-.tif', outputNum), 'Class', 'uint8');
csmu.volume2tif(baselineSim, ...
    sprintf('analysis\\%03d-sim-baseline-pattern.tif', outputNum), 'Class', 'uint8');
csmu.volume2tif(simPattern, ...
    sprintf('analysis\\%03d-sim-deconv-pattern.tif', outputNum), 'Class', 'uint8');

simPatternRange = csmu.range(simPattern, 'all')
refPatternRange = csmu.range(refPattern, 'all')

vSim = cat(4, simPattern, simPattern, simPattern);
vSim = permute(vSim, [1, 2, 4, 3]);
vBase = cat(4, baselineSim, baselineSim, baselineSim);
vBase = permute(vBase, [1, 2, 4, 3]);
vref = cat(4, refPattern, patternSpace, patternSpace);
vref = permute(vref, [1, 2, 4, 3]);
refBinMask = refPattern > 0.5;
refBinMask = permute(refBinMask, [1, 2, 4, 3]);
delta = vref - vSim;
opac = 25/100;
compositeSim = vSim + ((delta * opac) .* refBinMask);
delta = vref - vBase;
compositeBase = vBase + ((delta * opac) .* refBinMask);

csmu.volume2tif(compositeSim, ...
    sprintf('analysis\\%03d-sim-deconv-and-ref-pattern.tif', outputNum), ...
    'Class', 'uint8', ...
    'ColorDim', 3);

csmu.volume2tif(compositeBase, ...
    sprintf('analysis\\%03d-sim-baseline-and-ref-pattern.tif', outputNum), ...
    'Class', 'uint8', ...
    'ColorDim', 3);

duplicatedMask = maskSpace;
for iPage = 1:maskSpaceSize(3)
    duplicatedMask(:, :, iPage) = ...
        patternPageToMaskRes(...
        refPattern(:, :, backPatternSpacePage{iPage}));
end
duplicatedMask = double(duplicatedMask > 0.5);

addedPixels = (computedMask > 0.5) & ~(duplicatedMask > 0.5);
removedPixels = ~(computedMask > 0.5) & (duplicatedMask > 0.5);
bothPixels = (computedMask > 0.5) & (duplicatedMask > 0.5);
compositeMasks = cat(4, bothPixels, bothPixels, bothPixels) ...
    + cat(4, removedPixels, maskSpace, maskSpace) ...
    + cat(4, maskSpace, addedPixels, maskSpace);
compositeMasks = permute(compositeMasks, [1, 2, 4, 3]);

% vSim = cat(4, duplicatedMask, maskSpace, maskSpace);
% vSim = permute(vSim, [1, 2, 4, 3]);
% vref = cat(4, computedMask, computedMask, computedMask);
% vref = permute(vref, [1, 2, 4, 3]);
% refBinMask = computedMask > 0.5;
% refBinMask = permute(refBinMask, [1, 2, 4, 3]);
% delta = vref - vSim;
% opac = 85/100;
% compositeMasks = vSim + ((delta * opac) .* refBinMask);

csmu.volume2tif(compositeMasks, ...
    sprintf('analysis\\%03d-deconv-mask-changes.tif', outputNum), ...
    'Class', 'uint8', ...
    'ColorDim', 3);

%% define convolution functions
function pattern = forwardConv(mask, psf, patternSpace, ...
    forwardZSelArray, maskPageToComputeRes)
    tempOutput = zeros([size(patternSpace, [1, 2]), size(psf, 3)], ...
        'like', patternSpace);    
    pattern = patternSpace;
    for iMaskPlane = 1:size(mask, 3)
        tempOutput(:) = 0;
        maskPage = maskPageToComputeRes(mask(:, :, iMaskPlane));
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
    backwardZSelArray, patternPageToMaskRes, method)

    matchedMethod = ...
        validatestring(method, {'mean', 'page'}, ...
        'backwardConv', 'method');
    
    tempOutput = convn(pattern, psf, 'same');
    mask = maskSpace;
    for iMaskPlane = 1:size(maskSpace, 3)        
        [patternSpaceZSel, patternSpacePage] = ...
            backwardZSelArray{iMaskPlane, :};
        switch matchedMethod
            case 'mean'
                mask(:, :, iMaskPlane) = patternPageToMaskRes(...
                    mean(tempOutput(:, :, patternSpaceZSel), 3));
            case 'page'
                mask(:, :, iMaskPlane) = patternPageToMaskRes(...
                    tempOutput(:, :, patternSpacePage));
            otherwise
                error('Unexpected value for method recieved, "%s".', ...
                    method);
        end
    end
end

% downsampling functions
function Y = averagingDownsample(X, factor)
assert(all(csmu.isint(size(X) / factor)));
xSize = size(X, 2);
ySize = size(X, 1);
xSel = 1:factor:xSize;
ySel = 1:factor:ySize;

Y = zeros(size(X), 'like', X);
Y = reshape(Y, length(ySel), length(xSel), []);
i = 1;
for yshift = 1:factor
    for xshift = 1:factor
        Y(:, :, i) = X((ySel + yshift - 1), (xSel + xshift - 1));
        i =  i + 1;
    end
end
Y = mean(Y, 3);
end

function Y =  blurDownsample(X, xSel, ySel)
factor = size(X) / [length(ySel), length(xSel)];
sigma = factor / 4;
XPrime = imgaussfilt(X, sigma, 'FilterDomain', 'spatial');
Y = XPrime(ySel, xSel);
end