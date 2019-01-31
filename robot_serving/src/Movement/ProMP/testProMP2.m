close all;
clear;

% Demos
X = linspace(0, 1, 1 / 0.01)';
y1_y = 1.15 * X * 10;
y1_x = X .^2 .* sin(X * pi / 1.2) * 20;
y2_y = (0.95 * X) .^1.1 * 10;
y2_x = X .^1.2 .* sin((X + 0.03) * pi / 1.25) * 20;
y3_y = X .^1.1 * 10;
y3_x = (1.11 * X) .^1.25 .* sin(X * pi / 1.3) * 20;
y4_y = X .^1.11 * 10;
y4_x = X .* sin(X * pi / 1.11) * 20;

pmp = OriginalProMP([y1_x, y1_y, y2_x, y2_y, y3_x, y3_y, y4_x, y4_y], 2, .01, eye(2) * 0.0001);
pmp.build(LinearPhaseGenerator(), NormalizedGaussianBasisGenerator(10), false);
Y = pmp.mostProbable();

plot(Y(:, 2), Y(:, 3), 'k');
hold on;
grid on;
plot(y1_x, y1_y, '.');
plot(y2_x, y2_y, '.r');
plot(y3_x, y3_y, '.g');
plot(y4_x, y4_y, '.m');
axis([-5 18 -5 18])
legend('Y', 'y1', 'y2', 'y3', 'y4');
