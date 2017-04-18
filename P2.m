load('BlockFont.mat')
for ii = 1:10
    mem(:,ii) = reshape((font(:,:,ii)),[],1); % all memories as one-dimension inputs
end

userinput2 = input('Type the amount of numerals to be trained on: ');
userinput1 = input('Type an input in single quotes (x1, x2, x3, ..., x9, x10): ');

temp1 = str2double(userinput1(2:end));
temp2 = mem(:,temp1); % figures out what the input is


y = zeros(40,1); % y0
W = mem(:,1:userinput2)*mem(:,1:userinput2)'; % weight matrix
y = sign(W*y + temp2); % y1

for ii = 2:10 % amount of iterations
    y = sign(W*y); % y_ii
end

numeral = reshape(y,[8,5]);
figure(1);
subplot(1,2,1)
imagesc(reshape(temp2,[8,5])) % correct image
title('Correct image')
subplot(1,2,2)
imagesc(numeral) % the image of what network has coughed up
title('Trained Networks result')