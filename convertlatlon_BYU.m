% convert BYU map data to xy coordinates
clearvars; 

% load BYU map information
load('BYUdata.mat')

% plug in extent of map (m) (extpy same as python input)
extpy = 20;
ext = extpy*1000/2;
scale_factor = 111111;

% check that the ics are within the lat/lon limits of the map
flag = zeros(length(BYUdata.ics),1);
latlim = zeros(length(BYUdata.ics),2);
lonlim = zeros(length(BYUdata.ics),2);
for ii = 1:length(BYUdata.ics)
    iclat = BYUdata.ics(ii,1);
    iclon = BYUdata.ics(ii,2);
    findlat = BYUdata.find(ii,1);
    findlon = BYUdata.find(ii,2);
    latlim(ii,:) = [iclat - ext/scale_factor, iclat + ext/scale_factor];
    lonlim(ii,:) = [iclon - ext/(scale_factor), iclon + ext/(scale_factor)];
    
    % check to make sure find point is in the map limits
    if findlat < latlim(ii,2) && findlat > latlim(ii,1) && findlon < lonlim(ii,2) && findlon > lonlim(ii,1)
        flag(ii) = 1;
    end
    
end
sum(flag)

% create new struct of the map ics and coordinates
ind = find(flag==1);
BYUmap.LP = 'hiker';
BYUmap.extentkm = ext/1000;
BYUmap.sizekm = [num2str(2*ext/1000),'x',num2str(2*ext/1000)];
BYUmap.ics = BYUdata.ics(ind,:);
BYUmap.find = BYUdata.find(ind,:);
BYUmap.loadfilenames = BYUdata.loadfilenames(ind);
BYUmap.realtraj = BYUdata.realtraj(ind,:);
BYUmap.latlim = latlim(ind,:);
BYUmap.lonlim = lonlim(ind,:);

% convert latitude and longitude ics to body coordinates in map
szBW = 300*(ext/1000);
LLy = szBW;
LLx = szBW;
icsxy = [];
findxy = [];
for kk = 1:length(BYUmap.ics)
    iclat = BYUmap.ics(kk,1);
    iclon = BYUmap.ics(kk,2);
    findlat = BYUmap.find(kk,1);
    findlon = BYUmap.find(kk,2);
    ycrds = linspace(BYUmap.latlim(kk,1),BYUmap.latlim(kk,2),LLy);
    xcrds = linspace(BYUmap.lonlim(kk,1),BYUmap.lonlim(kk,2),LLx);
    iclonx = find(xcrds>=iclon-0.00005 & xcrds<=iclon+0.000005,1);
    iclaty = find(ycrds>=iclat-0.00005 & ycrds<=iclat+0.000005,1);
    findlonx = find(xcrds>=findlon-0.00005 & xcrds<=findlon+0.00005,1);
    findlaty = find(ycrds>=findlat-0.00005 & ycrds<=findlat+0.00005,1);
    auxic = [iclonx iclaty];
    auxfind = [findlonx findlaty];
    icsxy = [icsxy; auxic];
    findxy = [findxy; auxfind];
end
BYUmap.sizecells = [num2str(LLy),'x',num2str(LLx)];
BYUmap.dim = [LLy,LLx];
BYUmap.icsxy = icsxy;
BYUmap.findxy = findxy;

% convert real trajectories to body coordinates
trajxy = cell(length(BYUmap.realtraj),1);
for kk = 1:length(BYUmap.realtraj)
    trajlat = BYUmap.realtraj{kk}(:,1);
    trajlon = BYUmap.realtraj{kk}(:,2);
    ycrds = linspace(BYUmap.latlim(kk,1),BYUmap.latlim(kk,2),LLy);
    xcrds = linspace(BYUmap.lonlim(kk,1),BYUmap.lonlim(kk,2),LLx);
    for jj = 1:length(trajlat)
        trajlonx = find(xcrds>=trajlon(jj)-0.00005 & xcrds<=trajlon(jj)+0.00005,1);
        trajlaty = find(ycrds>=trajlat(jj)-0.00005 & ycrds<=trajlat(jj)+0.00005,1);

        trajxy{kk}(jj,:) = [trajlonx trajlaty];
    end
end
BYUmap.trajxy = trajxy;

% change name to match my other codes (not important)
mapBYU = BYUmap;

save(['mapdim',num2str(2*ext/1000),'BYU.mat'],'mapBYU')