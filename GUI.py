import tkinter as tk
from tkinter import filedialog
from app import *

ocr_codes_selected = False
ocr_first_question_selected = False
def select_directory():
    directory = filedialog.askdirectory()
    print("Selected directory:", directory)

    # Add your logic here to distinguish between Grades Sheet and Bubble images
    if selected_option.get() == "Grades Sheet":
        global ocr_codes_selected, ocr_first_question_selected
        ocr_codes_selected, ocr_first_question_selected = select_ocr()
        print("OCR for Codes selected:", ocr_codes_selected)
        print("OCR for First Question selected:", ocr_first_question_selected)

    if directory:  # If a directory is selected
        run_btn.pack(side=tk.BOTTOM, pady=20)  # Display the "Run" button
    else:
        run_btn.pack_forget()  # Hide the "Run" button if no directory is selected

def select_ocr():
    ocr = tk.IntVar()
    ocr_first_question_check = tk.Checkbutton(root, text="OCR for First Question", variable=ocr)
    ocr_first_question_check.pack(pady=5)

    return ocr_codes_var.get(), ocr.get()

def run_specific_function():
    # Call the function you want to execute when the "Run" button is clicked
    print("Run button clicked!")
    # Include the specific functionality you want to execute here

# Function called when Grades Sheet is selected
def grades_sheet_selected():
    select_directory()

# Function called when Bubble is selected
def bubble_selected():
    select_directory()

def GUI():
    # Create the main window
    root = tk.Tk()
    root.geometry("500x500")  # Size of the window
    root.title('Image Selection')

    # Options for the user to select
    options = ["Grades Sheet", "Bubble"]

    # Variable to store selected option
    selected_option = tk.StringVar(root)
    selected_option.set(options[0])  # Set default option

    # Dropdown menu to select between Grades Sheet and Bubble
    option_menu = tk.OptionMenu(root, selected_option, *options)
    option_menu.pack(pady=10)

    # Button to select directory
    select_dir_btn = tk.Button(root, text='Select Directory', command=select_directory)
    select_dir_btn.pack(pady=10)

    # Change background color of the window
    root.configure(bg='#FFD700')  # Use a hexadecimal color code for a golden yellow background

    # Button to run a specific function initially hidden
    run_btn = tk.Button(root, text='Run', command=lambda: get_samples_data(directory, ocr_codes_selected, ocr_first_question_selected), width=20)
    run_btn.pack_forget()  # Initially hide the "Run" button

    root.mainloop()

# Call the GUI function to run the application
GUI()
