"""Software behind the Capstone Project GUI."""
import tkinter as tk #Entire module
from tkinter import ttk #Used for styling the GUI
from tkinter import filedialog
from pipeline import obtain_data
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


LARGE_FONT = ("Verdana", 12)

def main():
    """Quick Testing Phase."""
    testObj = MyApp()
    testObj.mainloop()


class MyApp(tk.Tk):
    """Cancer Classification Data Product."""

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
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
            self.files_to_review.insert('end', name)
        self.files_to_review.grid(row=5, column=5, pady=10, padx=10)
    
    def predict(self, controller):
        """Trigger for the predict action."""
        data = list(map(obtain_data, self.filename))
        # Here the predictions will be made
        predictions = dict()
        controller.show_frame(MainDashboard)
        return predictions

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
        self.plot()
    
    def plot(self):
        """Plots the main graphics of the dashboard."""
        fig = Figure(figsize=(5,4), dpi=100)
        ax = fig.add_subplot()
        #Pseudo data included for example of one graphic
        label_sex = ["Male", "Female"]
        sizes = [98, 2]
        ax.pie(sizes, labels=label_sex)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=1, pady=10, padx=10)


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