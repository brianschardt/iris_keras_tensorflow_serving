# Train and Save Model on Iris data set using Keras
### Exported Model configured for Tensorflow Serving

There is no real example of a simple model that was trained using 
keras and then exported to in a format specifically for 
Tensorflow Serving. So for the sake of clarity I wanted to use the simplest 
example I could think of, which is the XOR logic gate. https://en.wikipedia.org/wiki/XOR_gate




### Getting Environment Set Up
Clone Repo
```angular2html
git clone https://github.com/brianalois/xor_keras_tensorflow_serving.git
cd xor_keras_tensorflow_serving
```

### PIPENV
#### Use Pipenv because we are awesome develoeprs
I am using pipenv in order to standardize environments, kind or like the famous NPM for node

###### Install Pipenv
https://docs.pipenv.org/
```angular2html
pip install pipenv
```
or if you are using mac install with homebrew
```angular2html
brew install pipenv
```

#### Install Dependencies
run this in the repo directory, installs files from Pipfile
```angular2html
pipenv install
```
#### Run it using pipenv
```angular2html
pipenv run python index.py
```
### No Pipenv
#### Don't want to use Pipenv because I am not awesome
If you do not want to use **pipenv** then you must install these dependencies
You must have tensorflow keras, and numpy installed(obviously)
```angular2html
pip install numpy
pip install tensorflow keras
```
run the file to export the trained model
```angular2html
python index.py
```
#### Variables

There are 2 variables starting at line 15

**model_version**: change this to change the 
name of the folder of the specific model version
```angular2html
model_version = "1"
```
**epoch**: the higher this number is the more accurate the model, but the longer it will take to train. 5000 is good, but may take a while
```angular2html
epoch = 100
```

