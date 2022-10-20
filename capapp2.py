"""Capstone Application
-----------------------

Contains the code for the GUI through which one uses
the machine learning model created using the models
module.
"""
import random
import tkinter as tk #Entire module
from tkinter import ttk #Used for styling the GUI
from tkinter import filedialog
from pipeline import extract_data, transform_data
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import os
from pandas import DataFrame
from numpy import histogram, asarray, argmax
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax
from pandastable import Table
from datetime import datetime

LARGE_FONT = ("Verdana", 12)

class gui(tk.Frame):
    """User interface class for classifying DICOM files.
    
    ...
    
    This application will control multiple
    pages and pass data throughout frames. This
    application will contain two total pages.
    The first page will allow the user to load
    the dicom files desired before making
    predictions. The second page will display
    the results of the predictions with the
    metadata from the gathered dicom files. The
    application will classify DICOM files related
    to mammograms or side images of breasts.

    Parameters
    ----------
    master : Tk
        The master window of the algorithm.
    
    """


    def __init__(self, master:tk.Tk,*args, **kwargs):
        """Initialize the GUI."""
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.modalities = {
            0:'MR',
            1:'CT',
            2:'PT',
            3:'MG'
        }
        self.sides = {
            0:'L',
            1:'R'
        }

        self.sex = {
            0:'F',
            1:'M'
        }
        self.class_names = {0:'Benign', 1:'Malignant'}
        self.model = load_model('tclass_V1')
        #Initial Settings
        self.master = master
        self.master.wm_title("The Capstone Project")
        self.master.minsize(900, 600)
        self.pack(side="top", fill="both", expand=True)
        self.grid_rowconfigure(5, weight=1)
        self.grid_columnconfigure(5, weight=1)
        self.data = list()
        self.filenames = tuple()
        self.start_page()
        self.startpage.tkraise()

    def start_page(self):
        """Page to load the user DICOM files.

        ...

        This page will be used by the user to direct
        the algorithm to the desired list of DICOM
        files that the user would like to have analyzed.
        The user will then press the predict button,
        causing the algorithm to call the trained model
        to make predictions upon the DICOM files selected.
        Once the predictions are finished, the second
        page will be displayed containing basic charts.
        """
        self.startpage = tk.Frame(self)
        self.startpage.grid(row=5, column=5, sticky="nsew")
        label = tk.Label(self.startpage, text="Please load the DICOM files.", font=LARGE_FONT)
        label.grid(row=1, column=5, pady=10, padx=10)

        cancel_button = tk.Button(self.startpage, text="Cancel", command=lambda: self.master.destroy())
        cancel_button.grid(row=9, column=1, pady=10, padx=10)
        load_button = tk.Button(self.startpage, text="Load Images", command=lambda: self.load_images())
        load_button.grid(row=9, column=5,pady=10, padx=10)
        predict_button = tk.Button(self.startpage, text="Predict", command=lambda: self.main_dashboard())
        predict_button.grid(row=9, column=9, pady=10, padx=10)
        self.startpage.grid_rowconfigure(5,weight=1)
        self.startpage.grid_columnconfigure(5, weight=1)

    def load_images(self):
        """Load Image Names
        
        ...
        
        Creates a list of paths directed towards
        the files of interests. These paths are then
        used within the predict function to load the
        data and make predictions based on said data.
        """
        self.files_to_review = tk.Listbox(self, width=100)
        self.filename = filedialog.askopenfilenames(initialdir="./", title="Select a File")
        for name in self.filename:
            self.files_to_review.insert('end', os.path.basename(name))
        self.files_to_review.grid(row=5, column=5, pady=10, padx=10)
        return self
    
    def predict(self):
        """Trigger for the predict action.
        
        ...
        
        Load the machine learning model and the related
        DICOM files to make predictions based on
        data found within the header file and image.
        It then saves the prediction together with
        the metadata on to the pandas `DataFrame`
        for reporting on the main dashboard page.
        The main dashboard page is finally loaded
        with all of the plots set on the top."""
        self.data = list(map(extract_data, self.filename))
        self.data = list(map(transform_data, self.data))
        self.data = DataFrame(self.data)
        predictions = self.model({'image': asarray(self.data['Image'].to_list()), 'cat':asarray(self.data[['age', 'side']])})
        if len(predictions) < 2 and len(predictions) > 0:
            self.predictions = predictions[0].numpy()
            self.data['score'] = [softmax(self.predictions).numpy()]
            self.data['pred_class'] = self.class_names[argmax(self.data['score'])]
        elif len(predictions) >= 2:
            self.predictions = predictions
            pred_data = list()
            for pred in self.predictions:
                score = softmax(pred)
                pclass = self.class_names[argmax(score)]
                pred_data.append({'score':score, 'pred_class':pclass})
            _df = DataFrame(pred_data)
            self.data = self.data.join(_df)
        self.maindashboard.tkraise()
        return self


    
    def main_dashboard(self):
        """Load the dashboard portion of the application.
        
        ...
        
        Loads the dashboard with six independent
        plots. Each chart will depend on the data
        provided and will only show when they have
        the data required to plot.
        """
        self.maindashboard = tk.Frame(self)
        self.maindashboard.grid(row=5, column=5, sticky="nsew")
        self.maindashboard.grid_rowconfigure(5, weight=1)
        self.maindashboard.grid_columnconfigure(5, weight=1)
        download_button = tk.Button(self.maindashboard, text="Download Report", command=lambda:self.download())
        download_button.grid(row=0, column=9, padx = 10, pady = 10, sticky='n')
        self.predict()
        self.plot()
        f = tk.Frame(self)
        f.grid(row=6, column=5, sticky='sew')
        pt = Table(f, dataframe=self.data.loc[:, self.data.columns != 'Image'])
        pt.show()
    
    def download(self):
        """Download a csv file containing basic data"""
        self.data.to_csv("predictions report {}.csv".format(datetime.now().strftime("%d-%m-%Y %H_%M_%S")), index=False)

    def plot(self):
        """Plot the main graphics of the dashboard
        
        ...

        Creates six charts comprised of three pie
        charts and three bar graphs. These are based
        on the predictions, sex of the patient, age,
        and the imaging modality.
        """
        fig = Figure(figsize=(9,9), dpi=100)
        fig.suptitle("Main Dashboard")
        #First Pie Chart
        ax1 = fig.add_subplot(231)
        self.data['sex'] = self.data['sex'].map(self.sex)
        label_sex = self.data['sex'].unique()
        sex_count = self.data['sex'].value_counts().to_list()
        ax1.pie(sex_count, labels=label_sex, autopct='%1.1f%%')
        ax1.set_title("Male-to-Female Ratio")
        #Second Pie Chart
        ax2 = fig.add_subplot(232)
        #pred_labels = ["Benign", "Malignant"] #Pseudo Data
        pred_counts = self.data['pred_class'].value_counts().to_list()
        pred_labels = self.data['pred_class'].unique()
        ax2.pie(pred_counts, labels=pred_labels, autopct='%1.1f%%')
        #Third Pie Chart
        ax3 = fig.add_subplot(233)
        self.data['modality'] = self.data['modality'].map(self.modalities)
        label_modality = self.data['modality'].unique()
        modality_count = self.data['modality'].value_counts().to_list()
        ax3.pie(modality_count, labels=label_modality, autopct='%1.1f%%')
        ax3.set_title("Modalities Used to Collect Data")
        #First Bar Graph
        ax4 = fig.add_subplot(234)
        ax4.hist(self.data['age'], bins='auto')
        ax4.set_title("Age of Patients")
        #Second Bar Graph
        ax5 = fig.add_subplot(235)
        ax5.set_title("Patient Sex")
        ax5.hist(self.data['sex'], bins='auto')
        #Third Bar Graph
        ax6 = fig.add_subplot(236)
        labels = ['Benign', 'Malignant']
        ax6.bar(pred_labels, pred_counts)
        ax6.set_title('Count of Classification')
        #Load the Chart on the GUI
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=5, pady=10, padx=10, sticky='ew')



root = tk.Tk()
root.title("TEST")
my_gui = gui(root)
root.mainloop()