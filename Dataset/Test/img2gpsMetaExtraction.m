% Specify the folder path
folderPath = './im2gps3ktest';

% Get a list of all image files in the folder
imageFiles = dir(fullfile(folderPath, '*.jpg')); % Change the file extension if needed

%print the number of images
disp(numel(imageFiles));

% Loop through each image file
for i = 1:numel(imageFiles)
    % Get the image file name
    imageName = imageFiles(i).name;
    
    % Read the image information using imfinfo()
    imageInfo = imfinfo(fullfile(folderPath, imageName));
    
    % Save the image information struct as a JSON file

    % Remove the file extension from the image name
    imageName = imageName(1:end-4);
    jsonFileName = [imageName, '.json'];
    jsonStr = jsonencode(imageInfo);
    fid = fopen(fullfile(folderPath, jsonFileName), 'w');
    fwrite(fid, jsonStr, 'char');
    fclose(fid);
end