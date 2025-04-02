from utils import Config
from pipelines import Predict

config = Config()
pred = Predict(config)

img_path = "test_samples/group13-6.jpg"
keywords = 'branch retinal artery occlusion (brao), pan-retinal photocoagulation (prp), hollenhorst plaque'
caption = pred.predict(img_path, keywords)

print(caption)