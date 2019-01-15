require 'image'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'

cutorch.setDevice(1)
batchSize = 32; --32 images of mitosis and non-mitosis each.
fileLocation = '../Training_Data/Gen_Dataset/mitosis/'
Folds = {'A03','A04','A05','A07','A10','A11','A12','A14','A15','A17','A18'}
local numOfImages_mit = torch.Tensor(#Folds)
local numOfImages_Nmit = torch.Tensor(#Folds)
local totalIm_mit = 0;
local totalIm_Nmit = 0;
for i=1,(#Folds) do
  local mfile = io.open('Filenames/'..Folds[i]..'_mit.txt')
  local nmfile = io.open('Filenames/'..Folds[i]..'_nmit.txt')
  numOfImages_mit[i] = 0;
  for _ in io.lines('Filenames/'..Folds[i]..'_mit.txt') do
  	numOfImages_mit[i] = numOfImages_mit[i]+1;
  	totalIm_mit = totalIm_mit +1;
  end
  numOfImages_Nmit[i] = 0;
  for _ in io.lines('Filenames/'..Folds[i]..'_nmit.txt') do
  	numOfImages_Nmit[i] = numOfImages_Nmit[i]+1;
  	totalIm_Nmit = totalIm_Nmit +1;
  end
end

local trNumOfImages = torch.Tensor(#Folds);
local allTrImages = 0;
for f=1,(#Folds) do
	if numOfImages_mit[f]>numOfImages_Nmit[f] then
		trNumOfImages[f] = numOfImages_Nmit[f]
		numOfImages_mit[f] = numOfImages_Nmit[f]
	else
		trNumOfImages[f] = numOfImages_mit[f]
		numOfImages_Nmit[f] = numOfImages_mit[f]
	end
	allTrImages = allTrImages + 2*trNumOfImages[f];
end
totalIm_mit = allTrImages/2;
totalIm_Nmit = allTrImages/2;

--local AllImages_mit = torch.Tensor(totalIm_mit,3,101,101);
--local AllImages_Nmit = torch.Tensor(totalIm_Nmit,3,101,101);
AllImages_mit = {};
AllImages_Nmit={};
local FoldIndex_mit = torch.Tensor(totalIm_mit);
local FoldIndex_Nmit = torch.Tensor(totalIm_Nmit);
local count_mit = 0;
local count_Nmit = 0;
for i=1,(#Folds) do
  print(i,numOfImages_mit[i],numOfImages_Nmit[i])
  local mfile = io.open('Filenames/'..Folds[i]..'_mit.txt')
  local nmfile = io.open('Filenames/'..Folds[i]..'_nmit.txt')
  local tempc=0;
  for l in io.lines('Filenames/'..Folds[i]..'_mit.txt') do
  	if tempc==numOfImages_mit[i] then break end
  	count_mit = count_mit +1;
  	tempc = tempc+1;
  	AllImages_mit[count_mit] = ('../Training_Data/Gen_Dataset/mitosis/'..l);
  	FoldIndex_mit[count_mit]=i;
  end
  tempc=0;
  for l in io.lines('Filenames/'..Folds[i]..'_nmit.txt') do
  	if tempc==numOfImages_Nmit[i] then break end
  	count_Nmit = count_Nmit +1;
  	tempc = tempc+1;
  	AllImages_Nmit[count_Nmit] = ('../Training_Data/Gen_Dataset/not_mitosis/'..l);
  	FoldIndex_Nmit[count_Nmit]=i;
  end
end


print(numOfImages_mit)
classes = {'mitosis','non-mitosis'}

--define networks here

D1 = nn.Sequential();
D1:add(nn.SpatialConvolution(3,16,2,2));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 50 X 50
D1:add(nn.SpatialConvolution(16,16,3,3));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 24 X 24
D1:add(nn.SpatialConvolution(16,16,3,3));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 11 X 11
D1:add(nn.SpatialConvolution(16,16,2,2));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 5 X 5
D1:add(nn.SpatialConvolution(16,16,2,2));
D1:add(nn.LeakyReLU(0.2,true));
D1:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 2 X 2
D1:add(nn.View(16*2*2));
D1:add(nn.Linear(16*2*2,100));
D1:add(nn.Linear(100,2));
D1:add(nn.LogSoftMax());

--print(D1);

--print(D1:forward(trainData.data[10]))

D2 = nn.Sequential();
D2:add(nn.SpatialConvolution(3,16,4,4));
D2:add(nn.LeakyReLU(0.2,true));
D2:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 49 X 49
D2:add(nn.SpatialConvolution(16,16,4,4));
D2:add(nn.LeakyReLU(0.2,true));
D2:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 23 X 23
D2:add(nn.SpatialConvolution(16,16,4,4));
D2:add(nn.LeakyReLU(0.2,true));
D2:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 10 X 10
D2:add(nn.SpatialConvolution(16,16,3,3));
D2:add(nn.LeakyReLU(0.2,true));
D2:add(nn.SpatialMaxPooling(2,2));
-- size: 16 X 4 X 4
D2:add(nn.View(16*4*4));
D2:add(nn.Linear(16*4*4,100));
D2:add(nn.Linear(100,2));
D2:add(nn.LogSoftMax());
--print(D2);

--cudnn.convert(D2, cudnn)
D2:cuda()
criterion = nn.ClassNLLCriterion():cuda()
trainer = nn.StochasticGradient(D2,criterion)
trainer.learningRate = 0.002
trainer.learningRateDecay = 1
trainer.shuffleIndices = 0
trainer.maxIteration = 20
--trainer:train(trainData)

for f=1,(#Folds) do
	local trSize = allTrImages - 2*(trNumOfImages[f]);
	local teSize = numOfImages_mit[f] + numOfImages_Nmit[f];
	--local trainImageAll = torch.Tensor(trSize,3,101,101)
	trainImageAll = {}
	local trainLabelAll = torch.Tensor(trSize)
	--local testImageAll = torch.Tensor(teSize,3,101,101)
	testImageAll = {}
	local testLabelAll = torch.Tensor(teSize)
	local trCount = 0;
	local teCount = 0;
	-- load mitosis data for all folds in train except fth fold and test for fth fold
	for j=1,(totalIm_mit) do
		if FoldIndex_mit[j]==f then
			teCount = teCount+1;
			testImageAll[teCount] = AllImages_mit[j];
			testLabelAll[teCount] = 1;
		else
			if trCount<= trSize/2 then
				trCount = trCount+1;
				trainImageAll[trCount] = AllImages_mit[j];
				trainLabelAll[trCount] = 1;
			end
		end
	end
	collectgarbage()
	print("mitosis loading done")
	-- load non-mitosis data for all folds in train except fth fold and test for fth fold
	for j=1,(totalIm_Nmit) do
		if FoldIndex_Nmit[j]==f then
			teCount = teCount+1;
			testImageAll[teCount] = AllImages_Nmit[j];
			testLabelAll[teCount] = 2;
		else
			if trCount<= trSize then
				trCount = trCount+1;
				trainImageAll[trCount] = AllImages_Nmit[j];
				trainLabelAll[trCount] = 2;
			end
		end
	end
	collectgarbage()
	print("non-mitosis loading done")
	
	print(trSize,teSize)
	print((#trainLabelAll)[1])

	local labelsShuffle = torch.randperm((#trainLabelAll)[1])

	local mean = {}
	local sum = {0,0,0}
	local std = {}
	local counter = 0;
	local im=torch.CudaTensor(3,101,101);
	for i=1,4 do --trSize do
		im = image.load(trainImageAll[i]);
		for c=1,3 do
			sum[c] = sum[c]+im[{c,{},{}}]:sum();
		end
		counter = counter+(101*101);
	end
	for c=1,3 do
		mean[c] = sum[c]/counter;
	end
	sum = {0,0,0}
	for i=1,4 do --trSize do
		im = image.load(trainImageAll[i]);
		for c=1,3 do
			im[{c,{},{}}]:add(-mean[c]);
			sum[c] = sum[c]+im[{c,{},{}}]:pow(2):sum();
		end
	end
	for c=1,3 do
		std[c] = torch.sqrt(sum[c]/counter);
	end
	print("preprocessing done :)");

	local iteration =1;
	local currentLearningRate = trainer.learningRate;
	local input=torch.CudaTensor(batchSize,3,101,101);
	local target=torch.CudaTensor(batchSize);
	while true do
		local currentError_ = 0
        for t = 1,(trSize/batchSize) do
        	local currentError = 0;
	      	for t1 = 1,batchSize do
	      		t2 = (t-1)*batchSize+t1;
	      		input[t1] = image.load(trainImageAll[labelsShuffle[t2]]);
	        	target[t1] = trainLabelAll[labelsShuffle[t2]]
	      		for i=1,3 do
				   -- normalize each channel globally:
				   input[{{},i,{},{}}]:add(-mean[i])
				   input[{{},i,{},{}}]:div(std[i])
				end
				currentError = currentError + criterion:forward(D2:forward(input[t1]), target[t1])
				currentError_ = currentError_ + currentError
         		D2:updateGradInput(input[t1], criterion:updateGradInput(D2:forward(input[t1]), target[t1]))
         		D2:accUpdateGradParameters(input[t1], criterion.gradInput, currentLearningRate)
	        end
	    end

		currentError_ = currentError_ / trSize
		print("#iteration "..iteration..": current error = "..currentError_);
    	iteration = iteration + 1
      	currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
      	if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
        	print("# StochasticGradient: you have reached the maximum number of iterations")
         	print("# training error = " .. currentError_)
         	break
      	end
	end


	torch.save("D2_batchFoldModel.t7",D2);
	--D2 = torch.load("D2_foldModel.t7")
	collectgarbage()



	testData = {
		data = {},
		labels = torch.Tensor(teSize),
		size = function() return teSize end
	}

	for i =1,teSize do
		testData.data[i] = testImageAll[i]
		testData.labels[i] = testLabelAll[i]
	end

	correct = 0
	class_perform = {0,0}
	class_size = {0,0}
	for i=1,teSize do
	    local groundtruth = testData.labels[i]
	    local example = image.load(testData.data[i])
	    for i=1,3 do
		   -- normalize each channel globally:
		   example[{ i,{},{} }]:add(-mean[i])
		   example[{ i,{},{} }]:div(std[i])
		end
	    --print('ground '..groundtruth)
	    class_size[groundtruth] = class_size[groundtruth] +1
	    local prediction = D2:forward(example:cuda())
	    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	    if groundtruth == indices[1] then
	        correct = correct + 1
	        class_perform[groundtruth] = class_perform[groundtruth] + 1
	    end
	end
	print(sys.COLORS.green.."Fold "..f.." done. Results ---->");
	print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
	for i=1,#classes do
	   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
	end
end
collectgarbage()