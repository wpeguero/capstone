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
from numpy import histogram

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

    def __init__(self, master:tk.Tk, *args, **kwargs):
        """Initialize the GUI."""
        tk.Frame.__init__(self, master, *args, **kwargs)
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
        """Load Image Names."""
        self.files_to_review = tk.Listbox(self)
        self.filename = filedialog.askopenfilenames(initialdir="/", title="Select a File")
        for name in self.filename:
            self.files_to_review.insert('end', os.path.basename(name))
        self.files_to_review.grid(row=5, column=5, pady=10, padx=10)
        return self
    
    def predict(self):
        """Trigger for the predict action."""
        self.data = list(map(extract_data, self.filename))
        self.data = list(map(transform_data, self.data))
        self.data = DataFrame(self.data)
        # Here the predictions will be made
        self.maindashboard.tkraise()
        
    def main_dashboard(self):
        self.maindashboard = tk.Frame(self)
        self.maindashboard.grid(row=5, column=5, sticky="nsew")
        label = tk.Label(self.maindashboard, text="Main Dashboard", font=LARGE_FONT)
        label.grid(row=1, column=5, pady=10, padx=10)
        self.maindashboard.grid_rowconfigure(5, weight=1)
        self.maindashboard.grid_columnconfigure(5, weight=1)
        self.predict()
        self.plot()
    
    def plot(self):
        """Plots the main graphics of the dashboard."""
        fig = Figure(figsize=(9,9), dpi=100)
        #First Pie Chart
        ax1 = fig.add_subplot(231)
        label_sex = ["Male", "Female"]
        sizes = [10, 90]
        ax1.pie(sizes, labels=label_sex, autopct='%1.1f%%')
        ax1.set_title("Male-to-Female Ratio")
        #Second Pie Chart
        ax2 = fig.add_subplot(232)
        pred_labels = ["Yes", "No"] #Pseudo Data
        pred_count = [65, 35]
        ax2.pie(pred_count, labels=pred_labels, autopct='%1.1f%%')
        #Third Pie Chart
        ax3 = fig.add_subplot(233)
        label_modality = ['MRI', 'CT', 'PT', 'Other']
        modality_count = [30, 40, 20, 10]
        ax3.pie(modality_count, labels=label_modality, autopct='%1.1f%%')
        ax3.set_title("Modalities Used to Collect Data")
        #First Bar Graph
        ax4 = fig.add_subplot(234)
        age_band = [0, 20, 30, 40, 50, 65, 70, 90, 120]
        counts, bins = histogram(self.data['age'])
        label_age = ['20-29', '30-39', '40-49', '50-59', '60-69']
        age_count = [3, 5, 8, 10, 6]
        ax4.hist(self.data['age'], bins='auto')
        ax4.set_title("Age of Patients")
        #Second Bar Graph
        ax5 = fig.add_subplot(235)
        ax5.set_title("Patient Weights")
        ax5.hist([130,140,120,220,190,160,150], bins='auto')
        #Third Bar Graph
        ax6 = fig.add_subplot(236)
        male_synth_age = random.sample(range(20,70), 20) #Synthetic data meant to simulate ages of males who have malignant tumors
        female_synth_age = random.sample(range(20, 70), 20) #Synthetic data meant to simulate ages of females who have malignant tumors
        ax6.bar(range(20),male_synth_age, label='men')
        ax6.bar(range(20),female_synth_age, label='women')
        ax6.set_title('Count of Positive Cases Based on Gender')
        #Load the Chart on the GUI
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=5, pady=10, padx=10)



if __name__ == "__main__":
    root = tk.Tk()
    root.title("TEST")
    my_gui = gui(root)
    root.mainloop()