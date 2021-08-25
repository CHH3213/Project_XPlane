% [time,a2,a3,a4,a5,a6,a7,a8,a9]=textread('Application Traffic Share by Hour.txt','%s%s%s%s%s%s%s%s%s'); %,'headerlines',4去掉头4行
% lengendstr={'Data and Browsing' 'Online Video' 'P2P File Sharing' 'Other File Sharing' 'Voice and Video Communications' 'Data Communications' 'Gaming Consoles' 'PC Gaming'};

% k = 1.4;
% a2 = [0:0.035714:1]'*k + 1/14*k + 0.6/14;
% a3 = 1-a2 + 1/14*k;



ap=1:1:14;
ae=[0.7, 1.7, 2.6, 3.7, 4.7, 5.5, 6.7, 7.6, 8.5 9.4, 10.4, 11.4, 12.5, 13.5];

p=polyfit(ae,ap,1);
aei=1:0.5:15;
api=polyval(p,aei);
% plot(t,y,'o',ti,yi,'-')

k = p(1);
a2 = [0:0.035714:1]'*k + 1/14*k + p(2)/14;
a3 = 1-a2 + 1/14*k;

a=[a2,a3];

[row column]=size(a);

data=zeros(29,2);

for i=1:1:row

    for j=1:1:column

%         data(i,j)=str2double(strtok(a(i,j),'%'));
        data(i,j)=a(i,j);
    end

end
data = data * 14;

ta=data/100;
% ta=data;
%颜色选择，最多提供11种颜色选择（依次是红、绿、蓝、紫红、青、紫、橙、棕、灰、黄、黑）

% color_a={[1 0 0];[0 1 0];[0 0 1];[1 0 1];[0 1 1];[0.6 0 1];[1 0.5 0];[0.5 0 0];[0.5 0.5 0.5];[1 1 0];[0 0 0]};
color_a={[1 0 1];[0 1 1]};
marker_b='*d^v><shpxo';%画图标志选择，最多提供11种

%colorarray={'r.' 'bo' 'g*' 'm+' 'c-' 'y:' };
figure(51)
hold on

% for i=1:1:column

%     plot(1:1:24,data(:,i),['-' marker_b(i)],'color',color_a{i});

% end

h=area([0:0.035714:1]'*14+1,data);

for i=1:1:2

    set(h(i),'FaceColor',color_a{i})

    set(h(i),'EdgeColor',color_a{i})

end
alpha(0.5)
axis([1 15 1 15])
axis square;
set(gca,'XTick',1:2:15,'FontSize',45)
set(gca,'YTick',1:2:15,'FontSize',45)
% maxdata=max(max(data));

plot([0:0.035714:1]'*14+1, [0:0.035714:1]'*14+1,'-k','LineWidth',2.0);
plot(aei,api,'--','Color',[0 0 1],'LineWidth',2.0)

hl23 = legend('Escape zone','Capture zone','$$a_p=a_e$$','Phase-transition line','FontSize',45);
set(hl23,'Box','on');
set(hl23,'interpreter','latex')
% set(hl23,'orientation','horizon')

plot(ae,ap,'o','Color',[0 0 1],'LineWidth',2.0)

txt1 = '$$a_p>a_e$$';
hl2=text(9,12,txt1,'FontSize',45);
set(hl2,'interpreter','latex')

% axis([1 20 0 20])
% set(gca,'XTick',1:1:15)
% set(gca,'XTickLabel',time);
% set(gca,'XTick',1:1:length(time))
% ylabelstr=cell(1,11);
% for i=1:1:11
% 
%     ylabelstr{i}=[num2str((i-1)*10) '%'];
% 
% end

% set(gca,'YTick',0:1:24)
% set(gca,'YTickLabel',ylabelstr);

hl21 = xlabel('$$a_e$$','FontName','Times New Roman','FontSize',45);
set(hl21,'interpreter','latex')
hl22 = ylabel('$$a_p$$','FontName','Times New Roman','FontSize',45,'Rotation',90);
set(hl22,'interpreter','latex')
% rotateticklabel(gca, 90);

% legend(lengendstr,'Location','BestOutside')

% exportfig(gcf,'classificationMatrixDataPlot.jpg', 'Format','jpeg','height',9.5,'width',25, ...

%     'Color','cmyk','Resolution',300, 'FontMode','Fixed','FontSize',12,'LineMode','Fixed','LineWidth',1); %生成.jpg图片

% exportfig(gcf,'classificationMatrixDataArea.jpg', 'Format','jpeg','height',9.5,'width',25, 'Bounds','loose', ... 'Color','cmyk','Resolution',300, 'FontMode','Fixed','FontSize',12,'LineMode','Fixed','LineWidth',1); %生成.jpg图片