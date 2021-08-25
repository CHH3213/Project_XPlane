% load("F:\ѵ������\policy_border83\rewards.mat")
clear;
clc;
load("rewards.mat")
for i=1:10000
    clear a
    a = episode_reward(1,i,:);
    a=cell2mat(a);
    a=sum(a);
    rewardS(i)=a;
end
for i=1:999 %���ݼ�Ȩƽ��
    y(i)=mean(rewardS(10*(i-1)+1:10*i+1));
end
y(1000)=mean(rewardS(9991:10000));
y1=y;

% for i=1:10000
%     clear a
%     a = episode_reward2(1,i,:);
%     a=cell2mat(a);
%     a=sum(a);
%     rewardS(i)=a;
% end
% for i=1:499
%     y(i)=mean(rewardS(10*(i-1)+1:10*i+1));
% end
% y(500)=mean(rewardS(4991:5000));
% y2=y;
load("reward1.mat")

for i=1:10000
    clear a
    a = episode_reward1(1,i,:);
    a=cell2mat(a);
    a=sum(a);
    rewardS(i)=a;
end
for i=1:999%���ݼ�Ȩƽ��
    y(i)=mean(rewardS(10*(i-1)+1:10*i+1));
end
y(1000)=mean(rewardS(9991:10000));
Y1=y;
for i=1:10000
    clear a
    a = episode_reward1(1,i,:);
    a=cell2mat(a);
    a=sum(a);
    rewardS(i)=a;
end
for i=1:999
    y(i)=mean(rewardS(10*(i-1)+1:10*i+1));
end
y(1000)=mean(rewardS(9991:5000));
Y2=y;

% ys=y1+y2;
ys=y1;
Ys=(Y1+Y2)*3;
% Ys=Y2;
z1 = smoothdata(ys,'gaussian',15);%���ݹ⻬
Z1 = smoothdata(Ys,'gaussian',15);

plot(ys(1:1000),'r')%��Ӱ�������
hold on
plot(Ys(1:1000),'b')
hold on
plot(z1(1:1000),'k','linewidth',4.5)%�������ͼ�����
hold on
plot(Z1(1:1000),'b--','linewidth',4.5)

set(gca,'FontSize',36)
legend('MADDPG-CBF','MADDPG')
axis([0,1000,-200,50])
xlabel('episodes/10')
ylabel('reward')
