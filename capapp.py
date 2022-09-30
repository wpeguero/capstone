"""Software behind the Capstone Project GUI."""
import tkinter as tk #Entire module
from tkinter import ttk #Used for styling the GUI
from tkinter import filedialog

def main():
    """Quick Testing Phase."""
    testObj = App()
    testObj.mainloop()


class App(tk.Tk):
    """Cancer Classification Data Product."""

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.wm_title("The Capstone Project")
        self.minsize(900,600)
        self.config(bg="skyblue")



if __name__ == "__main__":
    main()
