%%% Gets the edges of the map layers, while also plotting the edges with the ipp
%%% and find locations in 2 plots - one for XY and one for lat/lon coordinates
close all; clc; clearvars;
set(0,'DefaultFigureWindowStyle','docked');

tic
load('mapdim20BYU')
map = mapBYU;
filename = map.loadfilenames;
Elfall = cell(length(filename),1);
Einall = cell(length(filename),1);
Ilolaxy = cell(length(filename),1);
LFlolaxy = cell(length(filename),1);
%%
for ic = 1:length(filename)
    ic
    load(['BYUmaps/BW_LFandInac_Zelev_',filename{ic}])
    
    ycrds = linspace(map.latlim(ic,1),map.latlim(ic,2),map.dim(1));   %% switch 1 and 2 to flip y-axis
    xcrds = linspace(map.lonlim(ic,1),map.lonlim(ic,2),map.dim(1));
    
    %% extract all the edges of the BW matrices
    sZBW = size(BWLF);
    BWlf = flipud(BWLF);
    %     BWlf = BWLF;
    
    Elf=[];
    for ii=1:sZBW(1)
        for jj=1:sZBW(2)
            if BWlf(ii,jj) ==1
                Elf = [Elf; [ii,jj]];
            else
            end
        end
    end
    
    BWin = flipud(BWInac);
    %     BWin = BWInac;
    Ein=[];
    for ii=1:sZBW(1)
        for jj=1:sZBW(2)
            if BWin(ii,jj) ==1
                Ein = [Ein; [ii,jj]];
            else
            end
        end
    end
    Elfall{ic} = Elf;
    Einall{ic} = Ein;
    
    for n = 1:length(Ein)
        iclon = xcrds(Ein(n,2));
        iclat = ycrds(Ein(n,1));
        Ilolaxy{ic}(n,:) = [iclon iclat];
    end
    for n = 1:length(Elf)
        iclon = xcrds(Elf(n,2));
        iclat = ycrds(Elf(n,1));
        LFlolaxy{ic}(n,:) = [iclon iclat];
    end
    
    %% other map layers - copied over the temp layers from running in py
    Zelev = load(['BYUmaps/layers/',filename{ic},'/elv_data_temp.csv']);
    BWriver = load(['BYUmaps/layers/',filename{ic},'/rivers_data_temp.csv']);
    BWriverLF = load(['BYUmaps/layers/',filename{ic},'/rivers_bdd_data_temp.csv']);
    BWriverInac = load(['BYUmaps/layers/',filename{ic},'/rivers_bdd_inac_data_temp.csv']);
    BWroads = load(['BYUmaps/layers/',filename{ic},'/roads_data_temp.csv']);
    BWrroads = load(['BYUmaps/layers/',filename{ic},'/railroads_data_temp.csv']);
    BWpower = load(['BYUmaps/layers/',filename{ic},'/powerlines_data_temp.csv']);
    BWlakeLF = load(['BYUmaps/layers/',filename{ic},'/lakes_data_temp.csv']);
    BWlakeInac = load(['BYUmaps/layers/',filename{ic},'/lakes_inac_data_temp.csv']);
    BWtrails = load(['BYUmaps/layers/',filename{ic},'/trails_data_temp.csv']);
    allBWlayersnf = {BWriver; BWriverLF; BWroads; BWrroads; BWpower; BWlakeLF; BWtrails; BWriverInac; BWlakeInac};
    allBWlayers = {flipud(BWriver); flipud(BWriverLF); flipud(BWroads); flipud(BWrroads); flipud(BWpower); flipud(BWlakeLF); flipud(BWtrails); flipud(BWriverInac); flipud(BWlakeInac)};

    allEdges = cell(length(allBWlayers),1);
    for ij = 1:length(allBWlayers)
        BW = allBWlayers{ij};
        E=[];
        for ii=1:sZBW(1)
            for jj=1:sZBW(2)
                if BW(ii,jj) ==1
                    E = [E; [ii,jj]];
                else
                end
            end
        end
        allEdges{ij} = E;
    end
    Elayerlolaxy = cell(length(allEdges),1);
    for kk = 1:length(allEdges)
        auxE = allEdges{kk};
        for n = 1:length(auxE)
            iclon = xcrds(auxE(n,2));
            iclat = ycrds(auxE(n,1));
            Elayerlolaxy{kk}(n,:) = [iclon iclat];
        end
    end
    layers.name = {'BWriver'; 'BWriverLF'; 'BWroads'; 'BWrroads'; 'BWpower'; 'BWlakeLF'; 'BWtrails'; 'BWriverInac'; 'BWlakeInac'};
    layers.BW = allBWlayers;
    layers.edges = allEdges;
    layers.lolaxy = Elayerlolaxy;
    
    %% plot map layers (the LFs are separate from the inaccessible for plotting)
    figure(ic)
    % in x/y coordinates
    subplot(1,2,2)
    leg = cell(9,1);
    for ibw = 1:7
        if ~isempty(allEdges{ibw})
            bw = scatter(allEdges{ibw}(:,2),allEdges{ibw}(:,1),1,'.');
            daspect([1 1 1]), alpha(bw,0.9), hold on
            leg{ibw} = layers.name{ibw};
        else
        end
    end
    for ibw = 8:9
        if ~isempty(allEdges{ibw})
            inac = scatter(allEdges{ibw}(:,2),allEdges{ibw}(:,1),5,'filled','MarkerFaceColor','b','MarkerEdgeColor','b');
            alpha(inac,0.9),
            leg{ibw} = layers.name{ibw};
        else
        end
    end
    legend(leg(~cellfun('isempty',leg)))
    
    % in lat/lon coordinates
    subplot(1,2,1)
    for ibw = 1:7
        if ~isempty(Elayerlolaxy{ibw})
            bwl = scatter(Elayerlolaxy{ibw}(:,1),Elayerlolaxy{ibw}(:,2),1,'.');
            daspect([1 1 1]), alpha(bwl,0.9), hold on
        else
        end
    end
    for ibw = 8:9
        if ~isempty(Elayerlolaxy{ibw})
            inacl = scatter(Elayerlolaxy{ibw}(:,1),Elayerlolaxy{ibw}(:,2),5,'filled','MarkerFaceColor','b','MarkerEdgeColor','b');
            alpha(inacl,0.9),
        else
        end
    end
    legend(leg(~cellfun('isempty',leg)))
    
    %% plot edges (these are the total BW matrices, layers not separated)
    figure(ic)
    % in x/y coordinates
    Elf = Elfall{ic};
    Ein = Einall{ic};
    subplot(1,2,2)
    bwp = scatter(Elf(:,2),Elf(:,1),1,'.','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor',[0.5 0.5 0.5]);
    daspect([1 1 1])
    alpha(bwp,0.0009), hold on
    if ~isempty(Ein)
        inac = scatter(Ein(:,2),Ein(:,1),5,'filled','MarkerFaceColor','b');
        alpha(inac,0.9),
        %         legend('LFs','Inac')
    else
    end
    
    % in lat/lon coordinates
    Elfg = LFlolaxy{ic};
    Eing = Ilolaxy{ic};
    subplot(1,2,1)
    bwg = scatter(Elfg(:,1),Elfg(:,2),1,'.','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor',[0.5 0.5 0.5]);
    daspect([1 1 1])
    alpha(bwg,0.0009), hold on
    if ~isempty(Eing)
        inacg = scatter(Eing(:,1),Eing(:,2),5,'filled','MarkerFaceColor','b');
        alpha(inacg,0.9),
        %         legend('LFs','Inac')
    else
    end
    
    
    %% for BYU data: plot the lon/lat and x/y trajectories to compare and make sure they're the same
    subplot(1,2,1)
    plot(map.realtraj{ic}(:,2),map.realtraj{ic}(:,1),'.','markerfacecolor','k','markersize',5,'MarkerEdgeColor','k'), hold on
    subplot(1,2,2)
    plot(map.trajxy{ic}(:,1),map.trajxy{ic}(:,2),'.','markerfacecolor','k','markersize',5,'MarkerEdgeColor','k'), hold on
    
    %% plot the lon/lat and x/y ipp and find coords to compare and make sure they're the same
    subplot(1,2,1)
    plot(map.find(ic,2),map.find(ic,1),'gp','markerfacecolor','g','markersize',15,'MarkerEdgeColor','k'), hold on
    plot(map.ics(ic,2),map.ics(ic,1),'o','markerfacecolor','k','markersize',10,'MarkerEdgeColor','w')
    xlim(map.lonlim(ic,:)), ylim(map.latlim(ic,:))
    xlabel('longitude'), ylabel('latitude')
    axis square
    subplot(1,2,2)
    plot(map.findxy(ic,1),map.findxy(ic,2),'gp','markerfacecolor','g','markersize',15,'MarkerEdgeColor','k'), hold on
    plot(map.icsxy(ic,1),map.icsxy(ic,2),'o','markerfacecolor','k','markersize',10,'MarkerEdgeColor','w')
    xlim([0 map.dim(1)]), ylim([0 map.dim(1)])
    xlabel('longitude (x cells)'), ylabel('latitude (y cells)')
    axis square
    %     sgtitle(['IC ',map.key{ic}])
    sgtitle(['IC ',filename{ic}])
    
    %%
    set(gcf,'PaperPosition',[0,0,11,8],'paperorientation','landscape');
    %     print('-dpdf',['plots/BWcheck',num2str(ic),'.pdf'])
    %     save(['edgesK1.mat'],'Elfall','Einall','LFlolaxy','Ilolaxy')
    %     print('-dpdf',['plots/BYU/BWcheck_',filename{ic},'.pdf'])
%         save(['edgesBYU.mat'],'Elfall','Einall','LFlolaxy','Ilolaxy')
    save(['BYUlayeredges_',filename{ic}],'layers')
end

% save(['edgesBYU.mat'],'Elfall','Einall','LFlolaxy','Ilolaxy')
toc

%% check if IC is inaccessible
% ics = 6; % Koester 83, BYU 6
% icinac = zeros(ics,2);
% for iic = 1:ics
%     if ~isempty(Einall{iic,1})
%         [test,ind] = ismember([1500,1500],Einall{iic,1},'rows');
%         icinac(iic,:) = [iic, ind];
%     end
% end

