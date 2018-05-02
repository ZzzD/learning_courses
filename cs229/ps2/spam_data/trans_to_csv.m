myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'MATRIX.TEST'));
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  [spmatrix, tokenlist, Category] = readMatrix(fullFileName);
  trainMatrix = full(spmatrix);
  size(trainMatrix)
  dlmwrite(strcat(fullFileName, '.csv'), trainMatrix);
  dlmwrite(strcat(fullFileName, '_category.csv'), Category);
 end
 dlmwrite('tokenlist.csv', tokenlist);