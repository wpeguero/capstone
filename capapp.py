"""Software behind the Capstone Project GUI."""
import tkinter as tk #Entire module
from tkinter import ttk #Used for styling the GUI
from tkinter import filedialog
from pipeline import extract_data
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import os

LARGE_FONT = ("Verdana", 12)

def main():
    """Quick Testing Phase."""
    testObj = MyApp()
    testObj.mainloop()


class MyApp(tk.Tk):
    """Cancer Classification Data Product."""

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        #initial data
        self.data = list()
        self.wm_title("The Capstone Project")
        self.minsize(900,600)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Display the widgets
        self.frames = dict()
        for F in (StartPage, MainDashboard, DetailedDash):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        self.show_frame(StartPage)
    
    def show_frame(self, controller):
        """Display the page based on selection."""
        frame = self.frames[controller]
        frame.tkraise()


class StartPage(tk.Frame):
    """Page used to load the DICOM files."""

    def __init__(self, parent, controller):
        """Display the initial widgets."""
        tk.Frame.__init__(self, parent)
        self.data = list()
        self.filename = list()
        #Labels
        label = tk.Label(self, text="Please load the DICOM files.", font=LARGE_FONT)
        label.grid(row=1, column=5, pady=10, padx=10)
        #Buttons
        cancel_button = tk.Button(self, text="Cancel", command=lambda:controller.destroy())
        cancel_button.grid(row=9, column=1, pady=10, padx=10)
        load_button = tk.Button(self, text="Load Images", command=lambda: self.load_images())
        load_button.grid(row=9, column=5,pady=10, padx=10)
        predict_button = tk.Button(self, text="Predict", command=lambda: self.predict(controller))
        predict_button.grid(row=9, column=9, pady=10, padx=10)
        self.grid_rowconfigure(5,weight=1)
        self.grid_columnconfigure(5, weight=1)
    
    def load_images(self):
        """Load Image Names."""
        self.files_to_review = tk.Listbox(self)
        self.filename = filedialog.askopenfilenames(initialdir="/", title="Select a File")
        for name in self.filename:
            self.files_to_review.insert('end', os.path.basename(name))
        self.files_to_review.grid(row=5, column=5, pady=10, padx=10)
        return self
    
    def predict(self, controller):
        """Trigger for the predict action."""
        self.data = list(map(extract_data, self.filename))
        # Here the predictions will be made
        controller.predictions = dict()
        MainDashboard.plot(controller)
        controller.show_frame(MainDashboard)
        MainDashboard.plot(controller)

class MainDashboard(tk.Frame):
    """Page used to display the data in a dashboard format."""

    def __init__(self, parent, controller):
        """Display the initial widgets."""
        tk.Frame.__init__(self, parent)
        # Widgets
        label = tk.Label(self, text="Dashboard", font=LARGE_FONT)
        label.grid(row=1, column=5, pady=10, padx=10)
        self.grid_rowconfigure(5,weight=1)
        self.grid_columnconfigure(5, weight=1)
    
    @staticmethod
    def plot(self):
        """Plots the main graphics of the dashboard."""
        fig = Figure(figsize=(9,9), dpi=100)
        #First Pie Chart
        ax1 = fig.add_subplot(231)
        label_sex = ["Male", "Female"]
        print(self.data)
        sizes = self.data['sex']
        ax1.pie(sizes, labels=label_sex)
        #Second Pie Chart
        ax2 = fig.add_subplot(232)
        pred_labels = ["Yes", "No"] #Pseudo Data
        pred_count = [65, 35]
        ax2.pie(pred_count, labels=pred_labels)
        #Third Pie Chart
        ax3 = fig.add_subplot(233)
        label_modality = ['MRI', 'CT', 'PT', 'Other']
        modality_count = [30, 40, 20, 10]
        ax3.pie(modality_count, labels=label_modality)
        #First Bar Graph
        ax4 = fig.add_subplot(234)
        label_age = ['20-29', '30-39', '40-49', '50-59', '60-69']
        age_count = [3, 5, 8, 10, 6]
        ax4.bar(label_age, age_count)
        #Load the Chart on the GUI
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=5, pady=10, padx=10)


class DetailedDash(tk.Frame):
    """Page used to zoom in on the image data from the ."""

    def __init__(self, parent, controller):
        """Display the initial widgets."""
        tk.Frame.__init__(self, parent)
        # Widgets
        label = tk.Label(self, text="Dashboard", font=LARGE_FONT)
        label.grid(row=1, column=5, pady=10, padx=10)


if __name__ == "__main__":
    main()
#https://www.digitalocean.com/community/tutorials/tkinter-working-with-classes