function result = importmap_py(save_filename)
% import map layers from pycharm generated code


% whitehorn: [37.474015, -80.868333]
% nolin state park: [37.304158, -86.206131]
% kentland: [37.196809, -80.548387]

% fnameelev = '/Users/ah/PycharmProjects/ags_grabber/map_layers/elv_data_[37.474015, -80.868333]_10.0km.csv';
% fnameriv = '/Users/ah/PycharmProjects/ags_grabber/map_layers/rivers_data_[37.474015, -80.868333]_10.0km.csv';
% fnamerivbd = '/Users/ah/PycharmProjects/ags_grabber/map_layers/rivers_bdd_data_[37.474015, -80.868333]_10.0km.csv';
% fnamerivin = '/Users/ah/PycharmProjects/ags_grabber/map_layers/rivers_bdd_inac_data_[37.474015, -80.868333]_10.0km.csv';
% fnameroad = '/Users/ah/PycharmProjects/ags_grabber/map_layers/roads_data_[37.474015, -80.868333]_10.0km.csv';
% fnamerr = '/Users/ah/PycharmProjects/ags_grabber/map_layers/railroads_data_[37.474015, -80.868333]_10.0km.csv';
% fnamep = '/Users/ah/PycharmProjects/ags_grabber/map_layers/powerlines_data_[37.474015, -80.868333]_10.0km.csv';
% fnamelbd = '/Users/ah/PycharmProjects/ags_grabber/map_layers/lakes_data_[37.474015, -80.868333]_10.0km.csv';
% fnamelin = '/Users/ah/PycharmProjects/ags_grabber/map_layers/lakes_inac_data_[37.474015, -80.868333]_10.0km.csv';

% temp file names to read from
fnameelev = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\elv_data_temp.csv';
fnameriv = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\rivers_data_temp.csv';
fnamerivbd = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\rivers_bdd_data_temp.csv';
fnamerivin = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\rivers_bdd_inac_data_temp.csv';
fnameroad = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\roads_data_temp.csv';
fnamerr = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\railroads_data_temp.csv';
fnamep = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\powerlines_data_temp.csv';
fnamelbd = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\lakes_data_temp.csv';
fnamelin = 'C:\\Users\\Larkin\\ags_grabber\\map_layers\\lakes_inac_data_temp.csv';

Zelev = load(fnameelev);
BWriver = load(fnameriv);
BWriverLF = load(fnamerivbd);
BWriverInac = load(fnamerivin);
BWroads = load(fnameroad);
BWrroads = load(fnamerr);
BWpower = load(fnamep);
BWlakeLF = load(fnamelbd);
BWlakeInac = load(fnamelin);

sZBW = size(Zelev);

%% elevation smoothing
%%%%% SMOOTH gradient of elevation
sZ = size(Zelev);
sigma = 2;
sZelev = imgaussfilt(Zelev,sigma);
sZgrad = imgradient(sZelev,'CentralDifference');
BWs = edge(sZgrad,'canny',[0.01 0.3]);
BWelevationGrad = double(BWs);
% figure, imshow(BWs)

%% save BW matrices
BWLF = BWelevationGrad + BWriver + BWriverLF + BWlakeLF + BWroads + BWrroads + BWpower;
BWInac = BWriverInac + BWlakeInac;
BWLF(BWLF ~= 0) = 1;
BWInac(BWInac ~= 0) = 1;

file_path = 'C:\\Users\\Larkin\\ags_grabber\\matlab_data\\'; % define your own path here!
% save_filename is just the ics point in question
save(strjoin({file_path, 'BW_LFandInac_Zelev_',save_filename,'.mat'},''),'BWLF','BWInac','sZelev')

result = 0
% clearvars; clc; close all; % try cleaning things up
end