clear
load("E:\CHH3213_KING\研究生\导师\X-Plane\XPlane_mov\save\xplane_TD3_1\data\data.mat")

for i=1:400
reward=episode_reward(i);
reward=cell2mat(reward);
reward_eachSum=sum(reward);
reward_sum(i)=reward_eachSum;
end

for i=1:400
    action=apisode_action(1,i,:);
    action=cell2mat(action);
    [l,k] =size(action)

    action_1(:,i)=action(:,1);
    action_2(:,i)=action(:,2);
    action_3(:,i)=action(:,3);
    action_4(:,i)=action(:,4);     
    action_sum_1(i)=sum(action_1(i));
    action_sum_2(i)=sum(action_2(i));
    action_sum_3(i)=sum(action_3(i));
    action_sum_4(i)=sum(action_4(i));
end

plot(reward_sum,'k','linewidth',2.5)
set(gca,'FontSize',36)
legend('TD3-XPlane')
% axis([0,500,0,50000])
xlabel('episodes')
ylabel('reward')

figure;
plot(action_sum_1,'k','linewidth',2.5)
hold on;
plot(action_sum_2,'r','linewidth',2.5)
hold on;
plot(action_sum_3,'g','linewidth',2.5)
hold on;
plot(action_sum_4,'b','linewidth',2.5)

set(gca,'FontSize',36)
legend('TD3-XPlane')
% axis([0,500,0,50000])
xlabel('episodes')
ylabel('action')
