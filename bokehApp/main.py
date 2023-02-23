
from typing import Text
import numpy as np,os,shutil,csv,glob,time
from bokeh.layouts import column, row, Spacer,Row, Column, widgetbox
from bokeh.models import Button, Slider, Div, Select, TextInput, formatters
from bokeh.plotting import curdoc
from bokeh.colors import RGB
from bokeh.models import ColumnDataSource, TextInput, Button, Panel, Tabs, Label, DataTable, TableColumn
import pandas as pd
from bokeh.models import HTMLTemplateFormatter
import jinja2
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, output_file, show
from bokeh.models.widgets import FileInput
import numpy as np,sys,cv2,base64
from Model_scripts.Models import *
import tensorflow as tf
from sklearn import metrics
from bokeh.plotting import figure
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pandas as pd
from functools import partial
'''Paths to files and folders
================================'''
metric_filepath = 'bokehApp/Model_scripts/saved_models/metrics_scores.csv'



'''Call back Functions
================================
'''





#next button callback for WOT misclassified samples

#Function called when image is upload for prediction
def callback_Fileinput_imagePrediction(attr, old, new):
    imageStatus_TextTag.text = '<h3 style="text-align: center"></h3>'
    imageResult_TextTag.text = '<h3 style="text-align: center"> </h3>'
    global classifier_model_status
    if classifier_model_status == False:
        predictioncheck_modelloadstatus_TextTag.text = '<h2 style="text-align: center">Model is not in deployement state. Please load the model first!</h2>'
        return 
    result1 = get_imageprediction(file_input_image_prediction.value)
    imageResult_TextTag.text = '<h3 style="text-align: center">'+str(result1)+' </h3>'

#Loading model
def callback_loadmodel():
    modelLoadingButton.disabled=True
    modelloadingTextTag.text = '<h3 style="text-align: center">Model is being loaded.....</h3>'
    global classifier_model
    
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    physical_devices = tf.config.list_physical_devices('GPU') 

    tf.config.set_visible_devices(physical_devices[0:2], 'GPU')
    #model loading
    classifier_model = get_xception('test')
    classifier_model.load_weights('bokehApp/Model_scripts/saved_models/xception.h5')
        
    modelloadingTextTag.text = '<h3 style="text-align: center">Model and weights loaded..!!!</h3>'
    predictioncheck_modelloadstatus_TextTag.text = '<h2 style="text-align: center">Model is ready for prediction. Upload image to get Prediction!</h2>'
    global classifier_model_status
    classifier_model_status=True
    #return model

def callback_clearResulttextStatus():
    imageResult_TextTag.text = '<h3 style="text-align: center"> Result: .........</h3>'

#to stop the server
def callback_stopserver():
    sys.exit()


'''Utility functions
====================================
'''

def get_imageprediction(uploadedImage):
    global classifier_model
    imgdata1 = base64.b64decode(uploadedImage)
    #img1 = np.frombuffer(imgdata1, dtype='int32')
    filename1 = os.path.join('bokehApp','static','images','temp.jpg')  # I assume you have a way of picking unique filenames
    with open(filename1, 'wb') as f:
        f.write(imgdata1)
    img1 = cv2.imread(filename1)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(512,512))
    img = img/255.
    pred = classifier_model.predict(np.expand_dims(img,axis=0))
    a = pred[0].squeeze() < 0.5
    if a==True:
        confidence1 = round((((0.50-pred[0].squeeze()) / 0.50) *100 ),2)
        result_string = "Predicted class: NON-TUMOR" # with confidence scores "+str(confidence1)+'% '
    else:
        confidence1 = round((((pred[0].squeeze() - 0.50) / 0.50) *100 ),2)
        result_string = "Predicted class: TUMOR"# with confidence scores "+str(confidence1)+'% '
    #print(img1.shape)
    return result_string

'''
======================================
Widgets declaration & Global Variables
======================================'''

#Global variables
WT_misclassified_next_counter = 1
WOT_misclassified_next_counter = 1
classifier_model = None
classifier_model_status = False

#Model Loading
modelLoadingButton = Button(label="Load Model for Testing")
modelLoadingButton.on_click(callback_loadmodel)
modelloadingTextTag = Div(text='<h3 style="text-align: center">Model Not Loaded Yet!!!</h3>')

#Test loading text
testStatusTitle = Div(text='',width=300)

#file upload for prediction
file_input_image_prediction = FileInput(accept=".png,.jpg,.JPG")
file_input_image_prediction.on_change('value', callback_Fileinput_imagePrediction)
imageStatus_TextTag = Div(text='<p style="text-align: center">No Image uploaded</p>')
imageResult_TextTag = Div(text='<p style="text-align: center">Result: ---------</p>')
predictioncheck_modelloadstatus_TextTag = Div(text='<h2 style="text-align: center">Model is not ready for prediction!!!!</h2>')

#stop server button
stopserverButton = Button(label="Stop the server",align = 'end')
stopserverButton.on_click(callback_stopserver)

clearButton = Button(label="Clear")
clearButton.on_click(callback_clearResulttextStatus)
'''
===========================================================
Adding widgets into curdoc root to display on page
===========================================================
'''
#Graphical outline of the dashboard
curdoc().add_root(column(
                            row(stopserverButton),
                            
                            row(
                                Div(text = '<span style="padding-left:100px"></span>')
                            ),
                            row(Div(text = '<span style="padding-left:100px"></span>')),
                            
                            row(Div(text = "<h1 class='heading-title1' style='padding-left:500px;width:300px;' > Lungs Model Testing</h1>")),
                            row(modelLoadingButton  , modelloadingTextTag),
                            row(Div(text = '<span style="padding-left:100px"></span>')),
                            row(Div(text = '<span style="padding-left:100px"></span>')),

                            row(predictioncheck_modelloadstatus_TextTag),
                            #Test Block
                            row(column(
                                file_input_image_prediction,
                                imageStatus_TextTag
                                ),
                                column(clearButton)
                            ),
                            row(imageResult_TextTag),


                            #row(slider_layout),
                        )
                    )
'''           CSS styling custom
==========================================
'''
