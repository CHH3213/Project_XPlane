% reward = zeros(1,800);
% % reward_biaozhun = (61*60/2+500)/60
% for i = 1:800
%     reward(i) = sum(episode_reward{1,i});
% end
% episode = 1:800;
% reward_s = smoothdata(reward,'gaussian',30);


target =zeros(1,500);
target
reward = zeros(10,50);
reward_s = zeros(10,50);
reward_average = zeros(1,50);
reward_max = zeros(1,50);
reward_min = zeros(1,50);
for j = 1:10
    COUNT = num2str(j);
    nam1 = 'dataw';
    nam2 = num2str(j);
    nam3 = '.mat';
    filename = [nam1, nam2, nam3];
    load(filename);
    for i = 1:50
        reward(j,i) = sum(episode_reward{1,i});
    end
    reward_s(j,:)= smoothdata(reward(j,:),'gaussian',10);
    
end
episode = 1:50;
% reward_s = smoothdata(reward,'gaussian',15);

for i=1:50
    reward_average(1,i)=mean(reward(:,i)); %计算均值
        reward_max(1,i) = max(reward(:,i)); %计算最大值
    reward_min(1,i) = min(reward(:,i));%计算最小值
end

    
episode_conf = [episode episode(end:-1:1)];
% reward_conf = [reward_s+7 reward_s(end:-1:1)-7];
reward_average_conf=[ reward_max  reward_min(end:-1:1)];
figure(1);
p=fill(episode_conf,reward_average_conf,'g');
p.FaceColor= [175 238 238]/255
p.EdgeColor = 'none'
hold on 

for j = 1:10
    plot(episode,reward(j,:),'Color','#B0E0E6');hold on;
    plot(episode, reward_average,'k','linewidth',4,'Color','#48D1CC')%平均值后的图像绘制
%     plot(episode,reward_s(j,:),'k','linewidth',1,'Color','red')%滤波后的
end

    

% plot(reward_s(1:50),'k','linewidth',4,'Color','#FF9900')%美化后的图像绘制
% plot(reward(1:50),'k','linewidth',4,'Color','#FF9900')%美化后的图像绘制



set(gca,'FontName','Times New Roman','FontSize',45);
% set(gca,'FontSize',30);


hl21 = xlabel('Episode','FontName','Times New Roman','FontSize',50);
% set(hl21,'interpreter','latex')
hl22 = ylabel('Reward','FontName','Times New Roman','FontSize',50,'Rotation',90);
% set(hl22,'interpreter','latex')


% hl20 = legend('$$reward$$','FontSize',15);
% set(hl20,'Box','off');
% set(hl20,'interpreter','latex')




