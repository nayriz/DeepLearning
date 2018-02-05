from shutil import copyfile
import json


copyfile('/home/john/matterport/data_set/annotations/pascal_train2012.json', '/home/john/matterport/data_set/annotations/pascal_train2012_mod.json')
copyfile('/home/john/matterport/data_set/annotations/pascal_val2012.json', '/home/john/matterport/data_set/annotations/pascal_val2012_mod.json')


mapping = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]


# TRAINING SET
with open('/home/john/matterport/data_set/annotations/pascal_train2012_mod.json', 'r') as f:
    train = json.load(f)
    
    for i in range(len(train["annotations"])):        
        train["annotations"][i]["category_id"] = mapping[train["annotations"][i]["category_id"]-1]

with open('/home/john/matterport/data_set/annotations/pascal_train2012_mod.json', 'w') as f:
    f.write(json.dumps(train))


# VALIDATION SET
with open('/home/john/matterport/data_set/annotations/pascal_val2012_mod.json', 'r') as f:
    val = json.load(f)
    
    for i in range(len(val["annotations"])):        
        val["annotations"][i]["category_id"] = mapping[val["annotations"][i]["category_id"]-1]

with open('/home/john/matterport/data_set/annotations/pascal_val2012_mod.json', 'w') as f:
    f.write(json.dumps(val))


   
