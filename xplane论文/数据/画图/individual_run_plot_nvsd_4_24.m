% t=0:.001:4*pi;
% y=cos(t);
% hold on
% area(t,y)
% area(t(y>=0),y(y>=0),'FaceColor','r')

clear
clc

data = load('D:\weicloud\Research\Pursuit_Evasion_Project\policy_22\network_vs_default-a4_2.4.mat');
step = double(data.step)/10;
position = data.position;
velocity = data.volocity;
action = data.action_save;

agent_0_position = position(:,1:2);
agent_1_position = position(:,3:4);
agent_0_velocity = velocity(:,1:2);
agent_1_velocity = velocity(:,3:4);

agent_0_action = squeeze(action(:,1,[2 4]));
agent_1_action = squeeze(action(:,2,[2 4]));

agent_0_action_to_self = zeros(size(agent_0_action));
agent_1_action_to_self = zeros(size(agent_1_action));

agent_0_velocity_abs = sum(abs(agent_0_velocity).^2,2).^(1/2);
agent_1_velocity_abs = sum(abs(agent_1_velocity).^2,2).^(1/2);


% agent_0_action(1,:) * agent_0_velocity(1,:)'

for i =1:1:length(agent_0_action)
    theta = acos(agent_0_action(i,:)*agent_0_velocity(i,:)'/(norm(agent_0_action(i,:))*norm(agent_0_velocity(i,:))));
    v_vector = cross([agent_0_velocity(i,:),0],[agent_0_action(i,:),0]);
    if v_vector(1,3) < 0
%         v_vector(1,3)
        theta = -theta;
    end
    agent_0_action_to_self(i,1) = norm(agent_0_action(i,:))*cos(theta);
    agent_0_action_to_self(i,2) = norm(agent_0_action(i,:))*sin(theta);
end

for i =1:1:length(agent_1_action)
    theta = acos(agent_1_action(i,:)*agent_1_velocity(i,:)'/(norm(agent_1_action(i,:))*norm(agent_1_velocity(i,:))));
    v_vector = cross([agent_1_velocity(i,:),0],[agent_1_action(i,:),0]);
    if v_vector(1,3) < 0
%         v_vector(1,3)
        theta = -theta;
    end
    
    agent_1_action_to_self(i,1) = norm(agent_1_action(i,:))*cos(theta);
    agent_1_action_to_self(i,2) = norm(agent_1_action(i,:))*sin(theta);
end

agent_0_action_to_self = agent_0_action_to_self * 4;
agent_1_action_to_self = agent_1_action_to_self * 2.4;

% plot of position
figure(10);
hold on;
% plot([1:5:700],meanreward_to(1:5:700),'M-',[1:5:700],meanreward_te1(1:5:700),'B-',[1:5:700],meanreward_te2(1:5:700),'R-','LineWidth',1);
% plot([1:1:700],filtered_to(1:1:700),'M-',[1:1:700],filtered_te1(1:1:700),'B--',[1:1:700],filtered_te2(1:1:700),'R-.','LineWidth',2);
% plot(x1,y1,'B-.',x2,y2,'G--','LineWidth',2);
plot(agent_0_position(:,1),agent_0_position(:,2),'B-.','LineWidth',2);
plot(agent_1_position(:,1),agent_1_position(:,2),'--','Color',[0.19 0.50 0.08],'LineWidth',4);

% xlabel('Number of episodes','FontName','Times New Roman','FontSize',28);
% ylabel('Average reward','FontName','Times New Roman','FontSize',28);
hl2 = legend('$${x}_p(t)$$','$${x}_e(t)$$','FontSize',45);
set(hl2,'Box','off');
set(hl2,'interpreter','latex')
% legend('Hybrid DDPG strategy with \it{r_{w}}','End-to-end DDPG strategy with \it{r_{\tau1}}','End-to-end DDPG strategy with \it{r_{\tau2}}');
axis equal
axis([-5 5 -5 10])
set(gca,'FontSize',45);
% set(gca,'xtick',[0 100 200 300 400 500 600 700]);
% set(gca,'ytick',[-1.8 -1.6 -1.4 -1.2 -1.0 -0.8 -0.6 -0.4 -0.2 0]);

grid on;

figure(20);
hold on;
% plot([1:5:700],meanreward_to(1:5:700),'M-',[1:5:700],meanreward_te1(1:5:700),'B-',[1:5:700],meanreward_te2(1:5:700),'R-','LineWidth',1);
% plot([1:1:700],filtered_to(1:1:700),'M-',[1:1:700],filtered_te1(1:1:700),'B--',[1:1:700],filtered_te2(1:1:700),'R-.','LineWidth',2);
% plot(t,y1,'B-.',t,y2,'G--','LineWidth',2);
plot(step(1:length(agent_0_velocity_abs)),agent_0_velocity_abs(1:length(agent_0_velocity_abs),1),'B-.','LineWidth',2);
plot(step(1:length(agent_1_velocity_abs)),agent_1_velocity_abs(1:length(agent_1_velocity_abs),1),'--','Color',[0.19 0.50 0.08],'LineWidth',2);
axis square
hl21 = xlabel('$$t$$','FontName','Times New Roman','FontSize',45);
ylabel('Velocity','FontName','Times New Roman','FontSize',45);
hl20 = legend('$$||{\dot x}_p(t)||$$','$$||{\dot x}_e(t)||$$','FontSize',45);
set(hl20,'Box','off');
set(hl20,'interpreter','latex')
set(hl21,'interpreter','latex')

% legend('Hybrid DDPG strategy with \it{r_{w}}','End-to-end DDPG strategy with \it{r_{\tau1}}','End-to-end DDPG strategy with \it{r_{\tau2}}');
axis([0 length(agent_0_velocity_abs)*0.1 -1 8])
set(gca,'FontSize',45);
% set(gca,'xtick',[0 100 200 300 400 500 600 700]);
% set(gca,'ytick',[-1.8 -1.6 -1.4 -1.2 -1.0 -0.8 -0.6 -0.4 -0.2 0]);
grid on;

% figure(30);
% hold on;
% plot(step(1:200),agent_0_action_to_self(1:200,1),'B-.','LineWidth',1);
% plot(step(1:200),agent_1_action_to_self(1:200,1),'--','Color',[0.19 0.50 0.08],'LineWidth',2);
% axis square
% hl21 = xlabel('$$t$$','FontName','Times New Roman','FontSize',28);
% ylabel('Longitudinal acceleration','FontName','Times New Roman','FontSize',28);
% hl20 = legend('Pursuer','Evader','FontSize',28);
% set(hl20,'Box','off');
% set(hl20,'interpreter','latex')
% set(hl21,'interpreter','latex')
% 
% axis([0 20 -4.1 4.1])
% set(gca,'FontSize',28);
% grid on;
% 
% figure(31);
% hold on;
% plot(step(1:200),agent_0_action_to_self(1:200,2),'B-.','LineWidth',1);
% plot(step(1:200),agent_1_action_to_self(1:200,2),'--','Color',[0.19 0.50 0.08],'LineWidth',2);
% axis square
% hl21 = xlabel('$$t$$','FontName','Times New Roman','FontSize',28);
% ylabel('Lateral acceleration','FontName','Times New Roman','FontSize',28);
% hl20 = legend('Pursuer','Evader','FontSize',28);
% set(hl20,'Box','off');
% set(hl20,'interpreter','latex')
% set(hl21,'interpreter','latex')
% 
% axis([0 20 -4.1 4.1])
% set(gca,'FontSize',28);
% grid on;