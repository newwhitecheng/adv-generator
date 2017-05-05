#!/usr/bin/env th
require('torch')
require('nn')
require 'cutorch'
require('image')
require('paths')
torch.setdefaulttensortype('torch.FloatTensor')


local eye = 231              -- small net requires 231x231
--local label_nb = 364         -- label of 'bee'
local mean = 118.380948/255  -- global mean used to train overfeat
local std = 61.896913/255    -- global std used to train overfeat
local intensity = 3          -- pixel intensity for gradient sign

local modelPath = './model.net'
local srcListPath = './srcList_noGray.txt'
local destAdvListPath = './destList_noGray.txt'
local predResPath = './predRes.txt'


local function processOneImage(modelPath, srcImgPath, destAdvImgPath, predResfile)

	local path_img = srcImgPath
	local path_model = modelPath
	-- this script requires pretrained overfeat model from the repo
	-- (https://github.com/jhjin/overfeat-torch)
	-- it will fetch the repo and create the model if it does not exist


	-- get dependent files
	--if not paths.filep('model.net') then
   	--	os.execute([[
    --  		git clone https://github.com/jhjin/overfeat-torch
    --  		cd overfeat-torch
    --  		echo "torch.save('model.net', net)" >> run.lua
    --  		. install.sh && th run.lua && mv model.net .. && cd ..
   	--	]])
	--end
	--if not paths.filep('bee.jpg') then
   	--	os.execute('wget https://raw.githubusercontent.com/sermanet/OverFeat/master/samples/bee.jpg')
	--end
	--if not paths.filep('overfeat_label.lua') then
   	--	os.execute('wget https://raw.githubusercontent.com/jhjin/overfeat-torch/master/overfeat_label.lua')
	--end
	local label = require('overfeat_label')


	-- resize input/label
	local img = image.scale(image.load(path_img), '^'..eye)
	local tx = math.floor((img:size(3)-eye)/2) + 1
	local ly = math.floor((img:size(2)-eye)/2) + 1
	img = img[{{},{ly,ly+eye-1},{tx,tx+eye-1}}]
	img:add(-mean):div(std)

	-- get trained model (switch softmax to logsoftmax)
	local model = torch.load(path_model)
	model.modules[#model.modules] = nn.LogSoftMax()

	-- set loss function
	local loss = nn.ClassNLLCriterion()

	-- generate adversarial examples
	local img_adv, img_noise = require('adversarial-fast')(model, loss, img:clone(), label_nb, std, intensity)

    print (srcImgPath)
    predResfile:write(srcImgPath, '\n')

	model.modules[#model.modules] = nn.SoftMax()
	-- check prediction results
	local pred = model:forward(img)
	local val, idx = pred:max(pred:dim())
	print('==> original:', label[ idx[1] ], 'confidence:', val[1])
    predResfile:write('==> original:  ', label[ idx[1] ], '  confidence:', val[1], '\n')

	local pred = model:forward(img_adv)
	local val, idx = pred:max(pred:dim())
	print('==> adversarial:', label[ idx[1] ], 'confidence:', val[1])
    predResfile:write('==> adversarial:  ', label[ idx[1] ], '  confidence:', val[1], '\n')

	local img_diff = torch.add(img, -img_adv)
	print('==> mean absolute diff between the original and adversarial images[min/max]:', torch.abs(img_diff):mean(), '\n')
	predResfile:write('==> mean absolute diff between the original and adversarial images[min/max]:  ', torch.abs(img_diff):mean(), '\n')

	predResfile:write('\n')

	--image.save('img.png', img:mul(std):add(mean):clamp(0,255))
	image.save(destAdvImgPath, img_adv:mul(std):add(mean):clamp(0,255))
	--image.save('img_noise.png', img_diff:mul(127):clamp(0,255)

	--if pcall(require,'qt') then
  	--	local img_cat = torch.cat(torch.cat(img, img_adv, 3), img_diff:mul(127), 3)
  	--	image.display(img_cat)
	--end
end


local srcFile = io.open(srcListPath, 'r')
local destAdvFile = io.open(destAdvListPath, 'r')
local predResfile = io.open(predResPath, 'w')

local count = 1
while true do
      local srcImgPath = srcFile:read()
      if srcImgPath == nil then 
         break  
      end
      destAdvImgPath = destAdvFile:read()

      print (count)
      predResfile:write(count, '\n')

      processOneImage(modelPath, srcImgPath, destAdvImgPath, predResfile)
      count = count + 1
end

predResfile:close()
destAdvFile:close()
srcFile:close()
