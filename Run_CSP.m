load lake_data.mat;
for state = 1 : size(StudyRegion,1)
    num_cluster = 10;
    N = size(StudyRegion(state).DataAfterPca,1);
    % Compute the similarity matrix A (feature) using RBF kernel.Set diagonal entries to 0 for numerical consideration.
    dist_tmp = pdist(StudyRegion(state).DataAfterPca,'euclidean');
    sigma = median(dist_tmp);
    dist2_tmp = exp(-dist_tmp.^2/(2*sigma.^2));
    A=squareform(dist2_tmp);
    % Compute the graph Laplacian.
    sum_a=sum(A);
    vol = sum(sum_a);
    Dtmp=sum_a.^(-1/2);
    D_norm = diag(Dtmp);
    L=eye(N)-repmat(Dtmp',1,N).*A.*repmat(Dtmp,N,1); %L2 = eye(N) - D_norm*A*D_norm;
    % CSP
    U = csp_K (L, StudyRegion(state).NB, D_norm, vol, num_cluster);
    if any(isnan(U(:)))
        disp('U contains NaN');
        U(isnan(U)) = 0;
    end
    C = kmeans(U, num_cluster,'replicates',10,'MaxIter',200, 'start', 'cluster', ...
        'EmptyAction', 'singleton');
    Result(state,1) = struct('NumCluster',num_cluster,'ClassIdx',C);
end
save 'Run_CSP_result' 'Result'

%% visualize the result
my_color = jet(10);
for state = 1 : size(StudyRegion,1)
    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 1024, 768]);
    set(gcf,'Visible','on'); % not show the plot
    ax = usamap(StudyRegion(state).StateName);
    set(gcf,'color','white');
    
    lat = StudyRegion(state).LatLon(:,1);
    long = StudyRegion(state).LatLon(:,2);
    idx = Result(state).ClassIdx;
    num_cluster = Result.NumCluster;
    for i = 1:num_cluster
        geoshow(lat(idx==i),long(idx==i),'DisplayType','point','Marker','.','MarkerSize',15,'MarkerEdgeColor',my_color(i,:))% geoshow(lat,lon)
    end
    title(StudyRegion(state).StateName,'FontSize',15)
    picname = ['CSP_',StudyRegion(state).StateName];
    saveas(gcf,picname,'png');
    saveas(gcf,picname,'fig');
    close(FigHandle);
end








