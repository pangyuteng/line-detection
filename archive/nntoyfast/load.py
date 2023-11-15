from keras.models import model_from_yaml

yaml_file = 'nntoyfast.yml'
weight_file = 'checkpoint_weights_nntoyfast.h5'

with open(yaml_file,'r') as f:
    yaml_string = f.read()
                    
model = model_from_yaml(yaml_string)
model.load_weights(weight_file)
