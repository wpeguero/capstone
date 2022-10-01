"""Software behind the Capstone Project GUI."""
import tkinter as tk #Entire module
from tkinter import ttk #Used for styling the GUI
from tkinter import filedialog


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
        frame = StartPage(container, self)
        self.frames[StartPage] = frame
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
        # Widgets
        label = tk.Label(self, text="Please load the DICOM files.", font=LARGE_FONT)
        label.grid(row=1, column=5, pady=10, padx=10)
        cancel_button = tk.Button


if __name__ == "__main__":
    main()
#https://www.digitalocean.com/community/tutorials/tkinter-working-with-classes