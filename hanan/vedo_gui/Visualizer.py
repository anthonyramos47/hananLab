import os 
import sys

# Add hananLab to path
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

import vedo as vd 
import tkinter as tk
from tkinter import Entry, Label, Button, filedialog, Toplevel
import numpy as np
from hanan.optimization.Optimizer import Optimizer


class Visualizer():

    def __init__(self) -> None:
        # Create an Optimizer
        self.optimizer = None
        
        # Create a Vedo Plotter
        self.plotter = vd.Plotter(axes=1)
        
        # Create a Tkinter root window for the file dialog
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window
        self.opt_menu = None # Menu window

        self.constraints = None # Constraints dictionary {name: constraint}
        self.weights = None # List of weights
        self.iterations = 1 # Number of iterations

        self.meshes = [] # List of meshes

    def setup(self)-> None:
        """ Here we setup the geometry we are goint to optimize
        """
        pass

    def init_Visualizer(self) -> None:
        #self.constraints = constraints
        #self.optimizer = optimizer
        #self.weights = dict(zip(self.constraints.keys(), [0.0]*len(self.constraints)))
        self._init_menu()


    def init_menu(self) -> None:
        """ Here we initialize the menu as convenient
        """
        pass

    def _init_menu(self) -> None:
        """ Here we setup the basic initial menu buttons in Plotter
        """

        # Load mesh button
        self.plotter.add_button(self.load_mesh, 
                                           pos=(0.9, 0.9), 
                                           states=["Load Mesh"],
                                           c="w",
                                           bc="k",
                                           font="Calco",
                                           )

        # Optimizaiton button
        self.plotter.add_button(self.optimizations_settings, 
                                           pos=(0.2, 0.9), 
                                           states=["Optimization Settings"],
                                           c="w",
                                           bc="k",
                                           font="Calco",
                                           )


        self.init_menu()
        
        # # Show main plotter
        self.plotter.show(interactive=True)

    def display_weights(self) -> None:
        """ Function that displays the weights in plot window
        """

        for i, (name, value) in enumerate(self.weights.items()):
            t_w = vd.Text2D(f"Weight {name}: {value}", 
                                  pos=(0.05, 0.85-i*0.05),
                                  font="Calco",
                                  s=1.1,
                                  c="k",
                                  )
            self.plotter.add(t_w)
        self.plotter.render()

    def optimization_run(self) -> None:
        """ Function that runs the optimization
        """

        for _ in range(self.iterations):

            for name, constraint in self.constraints.items():
                constraint.set_weigth(self.weights[name])
                self.optimizer.get_gradients(constraint)
            
            self.optimizer.optimize()

            self.update_scene(self.optimizer.X)

    def reset_scene(self) -> None:
        """ Function that resets the scene
        """
        self.optimizer.reset()
        self.reset_scene(self.optimizer.X)
        pass

    def update_scene(self, X):
        """ Function that updates the scene, i.e. modify the meshes according to the variables X
        """
        pass

    def optimizations_settings(self) -> None:
        # setup the geometry
        self.setup()

        # Function to retrieve and display the entered values
        def run_opt():
            print("Entre")
            # Get the values from the Entry widgets and convert them to float
            entered_values = [float(entry.get()) for entry in entry_widgets]
            self.weights = dict(zip(self.weights.keys(), entered_values[:-1]))
            self.iterations = int(entered_values[-1])
            # Display the values in plotter
            self.display_weights()    
            
            #self.opt_menu.quit()
            self.opt_menu.destroy()

            # Run optimization
            self.optimization_run()

        self.opt_menu = Toplevel(self.root)
        self.opt_menu.title("Optimization Settings")


        # Create Entry widgets and labels dynamically
        entry_widgets = []
        for i, (name, value) in enumerate(self.weights.items()):
            
            label = Label(self.opt_menu, text=f"Enter {name}:")
            entry = Entry(self.opt_menu)
            entry.insert(0, str(value))  # Set default value in the Entry widget
            label.grid(row=i, column=0)
            entry.grid(row=i, column=1)
            entry_widgets.append(entry)

        # Define iterations entry
        iterations_label = Label(self.opt_menu, text=f"Iterations :")
        iterations_entry = Entry(self.opt_menu)
        iterations_entry.insert(0, str(self.iterations))  # Set default value in the Entry widget
        iterations_label.grid(row=len(self.weights), column=0)
        iterations_entry.grid(row=len(self.weights), column=1)
        entry_widgets.append(iterations_entry)

        # Create a button to trigger the input retrieval
        button_show = Button(self.opt_menu, text="Get Weights", command=run_opt)

        # Create a label to display the result
        label_result = Label(self.opt_menu, text="")

        # Arrange widgets using the grid layout
        button_show.grid (row=len(self.weights) + 1, columnspan=2)
        label_result.grid(row=len(self.weights) + 2, columnspan=2)

        self.root.mainloop()


    def show_meshes(self) -> None:
        """ Show the meshes
        """
        self.plotter.clear()
        for mesh in self.meshes:
            actor = vd.Mesh([mesh[0], mesh[1]])
            self.plotter.add(actor)
        
        self.plotter.render()


    def load_mesh(self) -> None:
        """ Load a mesh
        """
        file_dialog_options = {
        "title": "Select a File",
        "filetypes": [("Wave Font", "*.obj"), ("All Files", "*.*")]
        }
        file_path = filedialog.askopenfilename(**file_dialog_options)
        
        if file_path:
            actor = vd.load(file_path)
            self.meshes.append( [actor.points(), actor.faces()])
            self.plotter.add(actor)

        self.plotter.render()
 
