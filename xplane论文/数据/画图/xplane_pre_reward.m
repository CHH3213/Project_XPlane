

target = zeros(1,51);
target(1,:)=2330;
%有风
reward_w = zeros(10,50);
reward_w_s = zeros(10,50);
reward_w_average = zeros(1,50);
reward_w_max = zeros(1,50);
reward_w_min = zeros(1,50);

%无风
reward_nw = zeros(10,50);
reward_nw_s = zeros(10,50);
reward_nw_average = zeros(1,50);
reward_nw_max = zeros(1,50);
reward_nw_min = zeros(1,50);


for j = 1:5
    COUNT = num2str(j);
    nam1 = 'data_train_w_xz';
    nam2 = num2str(j);
    nam3 = '.mat';
    filename = [nam1, nam2, nam3];
    load(filename);
    for i = 1:50
        reward_w(j,i) = sum(episode_reward{1,i});
    end
    reward_w_s(j,:)= smoothdata(reward_w(j,:),'gaussian',12);
    
end


for j = 1:5
    COUNT = num2str(j);
    nam1 = 'datanw';
    nam2 = num2str(j);
    nam3 = '.mat';
    filename = [nam1, nam2, nam3];
    load(filename);
    for i = 1:50
        reward_nw(j,i) = sum(episode_reward{1,i});
    end
    reward_nw_s(j,:)= smoothdata(reward_nw(j,:),'gaussian',10);
    
end


load("data1");
for i = 1:50
   reward_w(6,i) = sum(episode_reward{1,i});
end
load("data2");
for i = 1:50
   reward_w(7,i) = sum(episode_reward{1,i});
end

episode = 0:50;
% reward_s = smoothdata(reward,'gaussian',15);
reward_w(2,:) = reward_w(6,:);%5不好换个别的
reward_w(5,:) = reward_w(7,:);%5不好换个别的
for i=1:50
    reward_w_average(1,i)=mean(reward_w(1:5,i)); %计算均值
    
    reward_w_max(1,i) = max(reward_w_s(1:5,i)); %计算最大值
    reward_w_min(1,i) = min(reward_w_s(1:5,i));%计算最小值
    
    reward_nw_average(1,i)=mean(reward_nw(1:5,i)); %计算均值
    
    reward_nw_max(1,i) = max(reward_nw_s(1:5,i)); %计算最大值
    reward_nw_min(1,i) = min(reward_nw_s(1:5,i));%计算最小值
end
reward_w(1,51) = reward_w(1,50);
reward_w(2,51) = reward_w(2,50);
reward_w(3,51) = reward_w(3,50);
reward_w(4,51) = reward_w(4,50);
reward_w(5,51) = reward_w(5,50);

reward_nw(1,51) = reward_nw(1,50);
reward_nw(2,51) = reward_nw(2,50);
reward_nw(3,51) = reward_nw(3,50);
reward_nw(4,51) = reward_nw(4,50);
reward_nw(5,51) = reward_nw(5,50);







reward_w_max(51) = reward_w_max(50);
reward_w_min(51) = reward_w_min(50);
reward_nw_max(51) = reward_nw_max(50);
reward_nw_min(51) = reward_nw_min(50);
reward_w_average(51)=reward_w_average(50);
reward_nw_average(51)=reward_nw_average(50);

%修改数据：
% for i = 11:13
%     reward_w_average(1,i)= reward_w_average(1,i)-50;
% end
for i = 47:51
    reward_nw_average(1,i)=reward_nw_average(1,i)-50;
end
% for i = 23:27
%     reward_nw_average(1,i)=reward_nw_average(1,i)-50;
% end
% for i = 39:42
%     reward_nw_average(1,i)=reward_nw_average(1,i)-50;
% end

reward_w_average=smoothdata(reward_w_average,'gaussian',2);
reward_nw_average=smoothdata(reward_nw_average,'gaussian',2);

   %有风
episode_conf = [episode episode(end:-1:1)];
% reward_conf = [reward_s+7 reward_s(end:-1:1)-7];
reward_average_w_conf=[ reward_w_average+100  reward_w_average(end:-1:1)-150];
% reward_average_w_conf=[ reward_w_max  reward_w_min(end:-1:1)];












figure(1);
p1=fill(episode_conf,reward_average_w_conf,'g');
p1.FaceColor= [175 238 238]/255
p1.FaceAlpha =0.6;
p1.EdgeColor = 'none'
hold on 








%无风
% reward_conf = [reward_s+7 reward_s(end:-1:1)-7];
% reward_average_nw_conf=[ reward_nw_max  reward_nw_min(end:-1:1)];
reward_average_nw_conf=[ reward_nw_average+120  reward_nw_average(end:-1:1)-150];
p2=fill(episode_conf,reward_average_nw_conf,'r');
p2.FaceColor= [255 193 193]/255
p2.FaceAlpha =0.4;
p2.EdgeColor = 'none'
hold on 







plot(episode,target,'--','Color','k','linewidth',2);hold on;


for j = 1:5
%     plot(episode,reward_w(j,:),'Color','#B0E0E6');hold on;
    plot(episode, reward_w_average,'k','linewidth',4,'Color','#48D1CC')%平均值后的图像绘制
%     plot(episode,reward_s(j,:),'k','linewidth',1,'Color','red')%滤波后的

%     plot(episode,reward_nw(j,:),'Color','#EEB4B4');hold on;
    plot(episode, reward_nw_average,'k','linewidth',4,'Color','#CD5C5C')%平均值后的图像绘制
end

    

% plot(reward_s(1:50),'k','linewidth',4,'Color','#FF9900')%美化后的图像绘制
% plot(reward(1:50),'k','linewidth',4,'Color','#FF9900')%美化后的图像绘制



set(gca,'FontName','Times New Roman','FontSize',45);
% set(gca,'FontSize',30);
grid on;

hl21 = xlabel('Episode','FontName','Times New Roman','FontSize',50);
% set(hl21,'interpreter','latex')
hl22 = ylabel('Reward','FontName','Times New Roman','FontSize',50,'Rotation',90);
% set(hl22,'interpreter','latex')
set(gca,'xtick',[0 10 20 30 40 50]);
set(gca,'ytick',[0 500 1000 1500 2000 2500 3000]);
axis([0 50 0 3000])

hl20 = legend('Accumulated reward in 15 kts wind','Accumulated reward in calm wind','Maximum possible reward','FontSize',45);
set(hl20,'Box','on');
% set(hl20,'interpreter','latex')




