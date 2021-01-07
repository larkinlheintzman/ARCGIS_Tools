file_names = {
    'map_layers\trails_data_temp.csv',
    'map_layers\roads_data_temp.csv',
    'map_layers\rivers_data_temp.csv',
    'map_layers\rivers_bdd_inac_data_temp.csv',
    'map_layers\rivers_bdd_data_temp.csv',
    'map_layers\railroads_data_temp.csv',
    'map_layers\powerlines_data_temp.csv',
    'map_layers\lakes_inac_data_temp.csv',
    'map_layers\lakes_data_temp.csv'
    };

figure(1); clf; hold on; grid on;

for i = 1:size(file_names,1)
    tmp = load(file_names{i});
    [r, c] = find(tmp);
    plot(c,3000-r,'.')
end
legend('trails','roads','rivers','rivers-bdd-inac','rivers-bdd','rails','powerlines','lakes-inac','lakes');


% file_path = 'punchbowl_track_meters.csv';
file_path = 'devilsditch_track_meters.csv';
track = load(file_path);
plot(track(1,:), track(2,:), 'k.', 'markersize',10)
plot(track(1,1),track(2,1),'gp','markersize',16,'markerfacecolor','g')
plot(track(1,end),track(2,end),'ko','markersize',16,'markerfacecolor','k')

axis equal