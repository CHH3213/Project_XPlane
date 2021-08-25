figure(10);
hold on;
roll_rate = episode_state{1,1255}(:,1);
pitch_rate = episode_state{1,1255}(:,2);
yaw_rate = episode_state{1,1255}(:,3);
roll = episode_state{1,1255}(:,4);
pitch = episode_state{1,1255}(:,5);
yaw = episode_state{1,1255}(:,6);
altitude = episode_state{1,1255}(:,7);
airspeed = episode_state{1,1255}(:,8);
step = 1:215;

plot(step,pitch_rate,'B-.','LineWidth',2);
plot(step,roll,'--','Color','#0072BD','LineWidth',4);
hl2 = legend('$${x}_p(t)$$','$${x}_e(t)$$','FontSize',15);
set(hl2,'Box','on');
set(hl2,'interpreter','latex')
axis equal
set(gca,'FontSize',15);
grid on;