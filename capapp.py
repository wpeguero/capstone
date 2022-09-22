"""Software behind the Capstone Project GUI."""
import tkinter as tk #Entire module
from tkinter import ttk #Used for styling the GUI
from tkinter import filedialog

def main():
    """Quick Testing Phase."""
    testObj = App()
    testObj.mainloop()


class App(tk.Tk):
    """Main window to show all apps from loading to image display post-prediction."""

    def __init__(self, *args, **kwargs):
        """Initialize the variable."""
        tk.Tk.__init__(self, *args, **kwargs)
        # Adding a title to the window
        self.wm_title("Test Application")

        # Creating a frame and assigning it to container
        container = tk.Frame(self, height=800, width=1600)
        # Specifying the region where the frame is packed in root
        container.pack(side="top", fill="both", expand=True)

        # Configuring the location of the container using grid
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # We will now create a dictionary of frames
        self.frames = {}
        # We'll create the frames themselves later but let's add the components to the dictionary
        for F in (MainPage, LoadingScreen, PredictionPage):
            frame = F(container, self)

            # The windows class acts as the root window for the frames.
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Using a method to switch frames
        self.show_frame(MainPage)

    def show_frame(self, cont):
        """Decides which frame to show based on specific trigger."""
        frame = self.frames[cont]
        # Raises the current frame to the top
        frame.tkraise()


class MainPage(tk.Frame):
    """The Main Page where one can view the observed files."""

    def __init__(self, parent, controller):
        """Initialize the windows."""
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Main Page")
        label.pack(padx=10, pady=10)

        # We use the switch_window_button in order to call the show_frame() method as a lambda function
        loading_button = tk.Button(
                self,
                text="Load Images",
                command=lambda: self.load_images()
                )
        loading_button.pack(side="bottom")

        predict_button = tk.Button(
                self,
                text="Predict",
                command=lambda: controller.show_frame(PredictionPage)
                )
        predict_button.pack(side="bottom")

    def load_images(self):
        """Load the image names."""
        self.filename = filedialog.askopenfilenames(initialdir="/", title="Select a File")



class LoadingScreen(tk.Frame):
    """Page where the user waits for the predictions to finish."""

    def __init__(self, parent, controller):
        """Start the loading screen on trigger."""
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="The predictions are still loading...")
        label.pack(padx=10, pady=10)
        switch_window_button = ttk.Button(
                self,
                text="Return to Menu",
                command=lambda: controller.show_frame(PredictionPage)
                )
        switch_window_button.pack(side="bottom", fill=tk.X)


class PredictionPage(tk.Frame):
    """Page containing the images post prediction."""

    def __init__(self, parent, controller):
        """Initialize display for predictions."""
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Completion Screen, we did it!")
        label.pack(padx=10, pady=10)
        switch_window_button = ttk.Button(
                self,
                text="Return to Menu",
                command=lambda: controller.show_frame(MainPage)
                )
        switch_window_button.pack(side="bottom", fill=tk.X)


if __name__ == "__main__":
    main()
