%% OD dynamics soil enrichment Mar 21
load('end_od.mat')
labels = {'pH 6.0','pH 7.3'};
for i = 1:2
    ods = squeeze(soil_end_od(2+i,[1 2 3 5 6 7],:)-median(soil_end_od(2+i,[4 8],:))); %blank subtract
    subplot(3,2,i)
    plot(1:size(soil_end_od,3),ods,'o-','LineWidth',2)
    ylim([-0.01 0.20])
    if i == 1
        ylabel('OD600')
    end
    xticks(1:size(soil_end_od,3))
    set(gca,'FontSize',16,'LineWidth',2)
    pbaspect([1.618 1 1])
    title(labels{i})
    xlim([0 12])
end

%%

load('nox_dynamics/processed_griess_data.mat')
t = [1,2,3,4,5,6,8,9,10,11,12,15,18,21,24,27,30,33];

blank_rows = [7,15];

ph6_no2 = 2.0*NO2(1:2:end,:)./median(NO3(blank_rows,:));
ph7_no2 = 2.0*NO2(2:2:end,:)./median(NO3(blank_rows+1,:));

ph6_no3 = 2.0*NO3(1:2:end,:)./median(NO3(blank_rows,:));
ph7_no3 = 2.0*NO3(2:2:end,:)./median(NO3(blank_rows+1,:));

%% plot metabolites at end of each cycle
tidx = find(mod(t,3)==0);
subplot(3,2,3)
plot(1:11,ph6_no3([1 2 3 5 6 7],tidx),'o-','LineWidth',2)
ylabel('NO_3^- (mM)')
set(gca,'FontSize',16,'LineWidth',2)
xlim([0 12])
ylim([0 2.3])
xticks(1:12)
pbaspect([1.618 1 1])

subplot(3,2,5)
plot(1:11,ph6_no2([1 2 3 5 6 7],tidx),'o-','LineWidth',2)
xlabel('cycle #')
ylabel('NO_2^- (mM)')
set(gca,'FontSize',16,'LineWidth',2)
xlim([0 12])
ylim([0 2.3])
xticks(1:12)
pbaspect([1.618 1 1])

subplot(3,2,4)
plot(1:11,ph7_no3([1 2 3 5 6 7],tidx),'o-','LineWidth',2)
set(gca,'FontSize',16,'LineWidth',2)
xlim([0 12])
ylim([0 2.3])
xticks(1:12)
pbaspect([1.618 1 1])

subplot(3,2,6)
plot(1:11,ph7_no2([1 2 3 5 6 7],tidx),'o-','LineWidth',2)
xlabel('cycle #')
set(gca,'FontSize',16,'LineWidth',2)
xlim([0 12])
ylim([0 2.3])
xticks(1:12)
pbaspect([1.618 1 1])
legend({'soil #1','soil #2','soil #3','soil #4','soil #5','soil #6'},'Orientation','horizontal')

set(gcf,'Position',[100 100 960 960])
print(gcf,'Fig_S_enrichment1_dynamics.png','-dpng','-r300')