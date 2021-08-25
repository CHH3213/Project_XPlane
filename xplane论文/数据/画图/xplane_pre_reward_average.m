% reward = zeros(1,800);
% % reward_biaozhun = (61*60/2+500)/60
% for i = 1:800
%     reward(i) = sum(episode_reward{1,i});
% end
% episode = 1:800;
% reward_s = smoothdata(reward,'gaussian',30);


%有风
reward_w = zeros(10,50);
reward_w_s = zeros(10,50);
reward_w_average = zeros(1,50);
reward_w_max = zeros(1,50);
reward_w_min = zeros(1,50);

%无风
reward_w = zeros(10,50);
reward_w_s = zeros(10,50);
reward_w_average = zeros(1,50);
reward_w_max = zeros(1,50);
reward_w_min = zeros(1,50);


for j = 1:10
    COUNT = num2str(j);
    nam1 = 'data';
    nam2 = num2str(j);
    nam3 = '.mat';
    filename = [nam1, nam2, nam3];
    load(filename);
    for i = 1:50
        reward_w(j,i) = mean(episode_reward{1,i});
    end
    reward_w_s(j,:)= smoothdata(reward_w(j,:),'gaussian',10);
    
end


for j = 1:10
    COUNT = num2str(j);
    nam1 = 'datanw';
    nam2 = num2str(j);
    nam3 = '.mat';
    filename = [nam1, nam2, nam3];
    load(filename);
    for i = 1:50
        reward_nw(j,i) = mean(episode_reward{1,i});
    end
    reward_nw_s(j,:)= smoothdata(reward_nw(j,:),'gaussian',10);
    
end
episode = 1:50;
% reward_s = smoothdata(reward,'gaussian',15);

for i=1:50
    reward_w_average(1,i)=mean(reward_w(:,i)); %计算均值
        reward_w_max(1,i) = max(reward_w_s(:,i)); %计算最大值
    reward_w_min(1,i) = min(reward_w_s(:,i));%计算最小值
    
        reward_nw_average(1,i)=mean(reward_nw(:,i)); %计算均值
        reward_nw_max(1,i) = max(reward_nw_s(:,i)); %计算最大值
    reward_nw_min(1,i) = min(reward_nw_s(:,i));%计算最小值
end

    
episode_conf = [episode episode(end:-1:1)];
% reward_conf = [reward_s+7 reward_s(end:-1:1)-7];
reward_average_w_conf=[ reward_w_max  reward_w_min(end:-1:1)];
figure(1);
p1=fill(episode_conf,reward_average_w_conf,'g');
p1.FaceColor= [175 238 238]/255
p1.FaceAlpha =0.6;
p1.EdgeColor = 'none'
hold on 

% reward_conf = [reward_s+7 reward_s(end:-1:1)-7];
reward_average_nw_conf=[ reward_nw_max  reward_nw_min(end:-1:1)];
p2=fill(episode_conf,reward_average_nw_conf,'r');
p2.FaceColor= [255 193 193]/255
p2.FaceAlpha =0.4;
p2.EdgeColor = 'none'
hold on 

for j = 1:10
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


hl21 = xlabel('Episode','FontName','Times New Roman','FontSize',50);
% set(hl21,'interpreter','latex')
hl22 = ylabel('Average Reward','FontName','Times New Roman','FontSize',50,'Rotation',90);
% set(hl22,'interpreter','latex')


hl20 = legend('reward_wind','reward_nowind','FontSize',45);
set(hl20,'Box','off');
set(hl20,'interpreter','latex')





