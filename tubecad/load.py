yaml_file = 'model.yml'
weight_file = 'weight.hdf5'

model.save_weights(weight_file)
yaml_string = model.to_yaml()
with open(yaml_file,'w') as f:
    f.write(yaml_string)
with open(yaml_file,'r') as f:
    loaded_yaml_string = f.read()
                    
from keras.models import model_from_yaml
model = model_from_yaml(loaded_yaml_string)
model.load_weights(weight_file)
