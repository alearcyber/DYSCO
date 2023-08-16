from BlurFilter import BlurFilter
import cv2
import TextureFilter
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ImageGrid import ImageGrid
from scipy.stats import chisquare


def test1():
    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg")
    expected = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/rawdisplay.png")
    f = BlurFilter(expected, observed, BlurFilter.GAUSSIAN, 15)
    f2 = BlurFilter(expected, observed, BlurFilter.MEDIAN, 31)
    cv2.imshow('original', f.o)
    cv2.imshow('gaussian blurred', f.o_filtered)
    cv2.imshow('median blur 5 x 5', f2.o_filtered)
    cv2.waitKey()

def test2():
    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg")
    out = TextureFilter.texture_features(observed)

    cv2.imshow('composite', np.mean(out, axis=2).astype(np.uint8))
    cv2.waitKey()
    for i in range(9):
        o = out[:, :, i]
        cv2.imshow('texture' + str(i), o)
    print(out.shape)
    cv2.waitKey()

#test the texture filter class itself
def test3():
    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg")
    tf = TextureFilter.TextureFilter(observed, observed)
    print(tf.o_filtered.shape)
    cv2.imshow('composite', tf.get_observed_composite())
    cv2.waitKey()

#testing some of the basic downsampling methods in opencv
def test4():
    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg")
    down = cv2.resize(observed, (11, 4) ,interpolation=cv2.INTER_AREA)
    out = cv2.resize(down, (200, 150), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('nearest', out)
    out = cv2.resize(down, (200, 150), interpolation=cv2.INTER_NEAREST_EXACT)
    cv2.imshow('nearest exact', out)

    down = cv2.resize(observed, (11, 4), interpolation=cv2.INTER_NEAREST_EXACT)
    out = cv2.resize(down, (200, 150), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('nearest 2', out)
    out = cv2.resize(down, (200, 150), interpolation=cv2.INTER_NEAREST_EXACT)
    cv2.imshow('nearest exact 2', out)
    cv2.waitKey()


def cvt_image(image):
    return ImageTk.PhotoImage(image=Image.fromarray(image))

def cvt2(image):
    one = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    two = cv2.resize(one, (450,200), interpolation=cv2.INTER_AREA)
    return two

def test5():
    #goal here is to have gui to show stuff
    #original -> blur -> downsample -> texture
    #dats it


    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png")
    expected = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/screenshot1.png")

    observed_np = np.copy(observed)
    expected_np = np.copy(expected)
    observed = cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)
    observed = cv2.resize(observed, (450,200), interpolation=cv2.INTER_AREA)
    expected = cv2.resize(cv2.cvtColor(expected, cv2.COLOR_BGR2RGB), (450,200), interpolation=cv2.INTER_AREA)


    #filters
    f2 = BlurFilter(expected_np, observed_np, BlurFilter.MEDIAN, 3)






    # Create an instance of tkinter window
    win = tk.Tk()

    # Define the geometry of the window
    win.geometry("1600x900")

    #scrollframe
    sf = tk.Canvas(win)
    sf.pack(fill=tk.BOTH, expand=tk.YES)

    #scrollbar
    scrollbar = tk.Scrollbar(win, command=sf.yview)


    frame = tk.Frame(sf, width=600, height=400)
    frame.pack(side=tk.LEFT)
    #frame.place(anchor='center', relx=0.5, rely=0.5)

    #description
    label3 = tk.Label(frame, text="Original")
    label3.pack(expand=True)


    #first image
    img = cvt_image(observed)  # Create an object of tkinter ImageTk
    label = tk.Label(frame, image=img)# Create a Label Widget to display the text or Image
    label.pack()

    #second image
    img2 = cvt_image(expected)
    label2 = tk.Label(frame, image=img2)
    label2.pack()




    #ROUNDTWO
    frame2 = tk.Frame(sf, width=600, height=400)
    frame2.pack(side=tk.LEFT)

    tk.Label(frame2, text="Median Blur").pack()
    tk.Label(frame2, image=cvt_image(cvt2(f2.o_filtered))).pack()
    tk.Label(frame2, image=cvt_image(cvt2(f2.e_filtered))).pack()



    scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

    #start
    win.mainloop()


def test6():
    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png")
    expected = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/screenshot1.png")
    observed = cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)
    observed = cv2.resize(observed, (800, 300), interpolation=cv2.INTER_AREA)
    expected = cv2.resize(cv2.cvtColor(expected, cv2.COLOR_BGR2RGB), (800, 300), interpolation=cv2.INTER_AREA)

    def populate(frame):
        '''Put in some fake data'''
        for col in range(5):
            # description
            tk.Label(frame, text="Original").grid(row=0, column=col)

            # first image
            img = cvt_image(observed)  # Create an object of tkinter ImageTk
            label = tk.Label(frame, image=img)  # Create a Label Widget to display the text or Image
            label.grid(row=1, column=col)

            # second image
            img2 = cvt_image(expected)
            label2 = tk.Label(frame, image=img2)
            label2.grid(row=2, column=col)
            """ 
            tk.Label(frame, text="%s" % row, width=3, borderwidth="1",
                     relief="solid").grid(row=row, column=0)
            t = "this is the second column for row %s" % row
            tk.Label(frame, text=t).grid(row=row, column=1)
            """
    def onFrameConfigure(canvas):
        '''Reset the scroll region to encompass the inner frame'''
        canvas.configure(scrollregion=canvas.bbox("all"))

    root = tk.Tk()
    canvas = tk.Canvas(root, borderwidth=0, background="#ffffff")
    frame = tk.Frame(canvas, background="#ffffff")
    vsb = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    canvas.configure(xscrollcommand=vsb.set)

    vsb.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((4, 4), window=frame, anchor="nw")

    frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

    populate(frame)

    root.mainloop()


def test7():
    root = tk.Tk()
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    frame_main = tk.Frame(root, bg="gray")
    frame_main.grid(sticky='news')

    label1 = tk.Label(frame_main, text="Label 1", fg="green")
    label1.grid(row=0, column=0, pady=(5, 0), sticky='nw')

    label2 = tk.Label(frame_main, text="Label 2", fg="blue")
    label2.grid(row=1, column=0, pady=(5, 0), sticky='nw')

    label3 = tk.Label(frame_main, text="Label 3", fg="red")
    label3.grid(row=3, column=0, pady=5, sticky='nw')

    # Create a frame for the canvas with non-zero row&column weights
    frame_canvas = tk.Frame(frame_main)
    frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw')
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    # Set grid_propagate to False to allow 5-by-5 buttons resizing later
    frame_canvas.grid_propagate(False)

    # Add a canvas in that frame
    canvas = tk.Canvas(frame_canvas, bg="yellow")
    canvas.grid(row=0, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=vsb.set)

    # Create a frame to contain the buttons
    frame_buttons = tk.Frame(canvas, bg="blue")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')

    # Add 9-by-5 buttons to the frame
    rows = 9
    columns = 5
    #buttons = [[tk.Button() for j in range(columns)] for i in range(rows)]
    buttons = [[tk.Label() for j in range(columns)] for i in range(rows)]
    images = []
    k = -1
    for i in range(0, rows):
        for j in range(0, columns):
            #buttons[i][j] = tk.Button(frame_buttons, text=("%d,%d" % (i + 1, j + 1)))
            #buttons[i][j].grid(row=i, column=j, sticky='news')
            #buttons[i][j] = tk.Label(frame_buttons, text="gridlabel", fg="green")
            #read in image
            observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png")
            observed = cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)
            observed = cv2.resize(observed, (100, 30), interpolation=cv2.INTER_AREA)
            observed = ImageTk.PhotoImage(image=Image.fromarray(observed))
            images.append(observed)

            buttons[i][j] = tk.Label(frame_buttons, image=images[-1])
            buttons[i][j].grid(row=i, column=j, sticky='news')

    # Update buttons frames idle tasks to let tkinter calculate buttons sizes
    frame_buttons.update_idletasks()

    # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
    first5columns_width = sum([buttons[0][j].winfo_width() for j in range(0, columns)])
    first5rows_height = sum([buttons[i][0].winfo_height() for i in range(0, columns)])
    frame_canvas.config(width=first5columns_width + vsb.winfo_width(),
                        height=first5rows_height)

    # Set the canvas scrolling region
    canvas.config(scrollregion=canvas.bbox("all"))

    # Launch the GUI
    root.mainloop()


#same as test 7, but I assign a variable for the image to the object
def test8():
    root = tk.Tk()
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    frame_main = tk.Frame(root, bg="gray")
    frame_main.grid(sticky='news')

    label1 = tk.Label(frame_main, text="Label 1", fg="green")
    label1.grid(row=0, column=0, pady=(5, 0), sticky='nw')

    label2 = tk.Label(frame_main, text="Label 2", fg="blue")
    label2.grid(row=1, column=0, pady=(5, 0), sticky='nw')

    label3 = tk.Label(frame_main, text="Label 3", fg="red")
    label3.grid(row=3, column=0, pady=5, sticky='nw')

    # Create a frame for the canvas with non-zero row&column weights
    frame_canvas = tk.Frame(frame_main)
    frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw')
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    # Set grid_propagate to False to allow 5-by-5 buttons resizing later
    frame_canvas.grid_propagate(False)

    # Add a canvas in that frame
    canvas = tk.Canvas(frame_canvas, bg="yellow")
    canvas.grid(row=0, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=vsb.set)

    # Create a frame to contain the buttons
    frame_buttons = tk.Frame(canvas, bg="gray")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')


    #headings


    # Add 9-by-5 buttons to the frame
    rows = 9
    columns = 2
    # buttons = [[tk.Button() for j in range(columns)] for i in range(rows)]
    buttons = [[tk.Label() for j in range(columns)] for i in range(rows)]
    for i in range(0, rows):
        for j in range(0, columns):
            # buttons[i][j] = tk.Button(frame_buttons, text=("%d,%d" % (i + 1, j + 1)))
            # buttons[i][j].grid(row=i, column=j, sticky='news')
            # buttons[i][j] = tk.Label(frame_buttons, text="gridlabel", fg="green")
            # read in image
            observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png")
            observed = cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)
            observed = cv2.resize(observed, (700, 280), interpolation=cv2.INTER_AREA)
            observed = ImageTk.PhotoImage(image=Image.fromarray(observed))
            buttons[i][j] = tk.Label(frame_buttons, image=observed)
            buttons[i][j].image = observed
            buttons[i][j].grid(row=i, column=j, sticky='news', padx=10, pady=10)

    # Update buttons frames idle tasks to let tkinter calculate buttons sizes
    frame_buttons.update_idletasks()

    # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
    first5columns_width = sum([buttons[0][j].winfo_width() for j in range(0, columns)])
    first5rows_height = sum([buttons[i][0].winfo_height() for i in range(0, columns)])
    frame_canvas.config(width=first5columns_width + vsb.winfo_width(),
                        height=first5rows_height)

    # Set the canvas scrolling region
    canvas.config(scrollregion=canvas.bbox("all"))

    # Launch the GUI
    root.mainloop()



def test9():

    #setting up the tkinter gui
    root = tk.Tk()
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frame_main = tk.Frame(root, bg="gray")
    frame_main.grid(sticky='news')

    #column headers at the top
    label3 = tk.Label(frame_main, text="Info", fg="blue", bg='white')
    label3.grid(row=0, column=0, pady=5, sticky='')
    label1 = tk.Label(frame_main, text="Observed", fg="blue", bg='white')
    label1.grid(row=0, column=1, pady=(5, 0), sticky='')
    label2 = tk.Label(frame_main, text="Expected", fg="blue", bg='white')
    label2.grid(row=0, column=2, pady=(5, 0), sticky='')


    # Create a frame for the canvas with non-zero row&column weights
    frame_canvas = tk.Frame(frame_main)
    frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw', columnspan=3)
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    # Set grid_propagate to False to allow 5-by-5 buttons resizing later
    frame_canvas.grid_propagate(False)

    # Add a canvas in that frame
    canvas = tk.Canvas(frame_canvas, bg="yellow")
    canvas.grid(row=0, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=vsb.set)

    # Create a frame to contain the buttons
    frame_buttons = tk.Frame(canvas, bg="gray")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')



    # Add 9-by-5 buttons to the frame
    rows = 9
    columns = 3
    xpadding = 10
    ypadding = 10
    # buttons = [[tk.Button() for j in range(columns)] for i in range(rows)]
    buttons = [[tk.Label() for j in range(columns)] for i in range(rows)]

    messages = []

    for i in range(0, rows):
        description = tk.Message(frame_buttons,text="Description of the images here")
        description.grid(row=i, column=0, padx=5, pady=5)
        messages.append(description)
        description
        for j in range(1, columns):
            # buttons[i][j] = tk.Button(frame_buttons, text=("%d,%d" % (i + 1, j + 1)))
            # buttons[i][j].grid(row=i, column=j, sticky='news')
            # buttons[i][j] = tk.Label(frame_buttons, text="gridlabel", fg="green")
            # read in image
            observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png")
            observed = cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)
            observed = cv2.resize(observed, (700, 280), interpolation=cv2.INTER_AREA)
            observed = ImageTk.PhotoImage(image=Image.fromarray(observed))
            buttons[i][j] = tk.Label(frame_buttons, image=observed)
            buttons[i][j].image = observed
            buttons[i][j].grid(row=i, column=j, sticky='news', padx=xpadding, pady=ypadding)

    # Update buttons frames idle tasks to let tkinter calculate buttons sizes
    frame_buttons.update_idletasks()

    # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
    first5columns_width = sum([buttons[0][j].winfo_width()+(xpadding*2) for j in range(0, columns)]) + 100


    first5rows_height = sum([buttons[i][0].winfo_height()+(ypadding*2) for i in range(0, columns)])
    first5rows_height = 900
    frame_canvas.config(width=first5columns_width + vsb.winfo_width(),
                        height=first5rows_height)

    # Set the canvas scrolling region
    canvas.config(scrollregion=canvas.bbox("all"))

    # Launch the GUI
    root.mainloop()


def build_gui_prototype():
    #setup some test data
    images = [
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail1.jpg'),#0
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail2.jpg'), #1
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg'), #2
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail4.jpg'), #3
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail6.jpg'), #4
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png'), #5
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/dash.png'), #6
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/dash4.jpg'), #7
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/rawdisplay.png'), #8
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/SlicTest.png'), #9
    ]
    data = []
    for i in range(0, 9, 2):
        data.append((f'this is images {i+1} and {i+2}', images[i], images[i+1]))



    # setting up the tkinter gui
    root = tk.Tk()
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frame_main = tk.Frame(root, bg="gray")
    frame_main.grid(sticky='news')

    # column headers at the top
    label3 = tk.Label(frame_main, text="Info", fg="blue", bg='white')
    label3.grid(row=0, column=0, pady=5, sticky='')
    label1 = tk.Label(frame_main, text="Observed", fg="blue", bg='white')
    label1.grid(row=0, column=1, pady=(5, 0), sticky='')
    label2 = tk.Label(frame_main, text="Expected", fg="blue", bg='white')
    label2.grid(row=0, column=2, pady=(5, 0), sticky='')

    # Create a frame for the canvas with non-zero row&column weights
    frame_canvas = tk.Frame(frame_main)
    frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw', columnspan=3)
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    # Set grid_propagate to False to allow 5-by-5 buttons resizing later
    frame_canvas.grid_propagate(False)

    # Add a canvas in that frame
    canvas = tk.Canvas(frame_canvas, bg="yellow")
    canvas.grid(row=0, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=vsb.set)

    # Create a frame to contain the buttons
    frame_buttons = tk.Frame(canvas, bg="gray")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')

    # setup frame
    rows = len(data)
    columns = 3
    xpadding = 10
    ypadding = 10
    buttons = [[tk.Label() for j in range(columns)] for i in range(len(data))]
    messages = []
    the_images_for_some_reason = []

    for i in range(0, rows):
        #get the data for this transformation
        desc, i1, i2 = data[i]

        #create description label
        description = tk.Message(frame_buttons, text=desc)
        description.grid(row=i, column=0, padx=5, pady=5)
        messages.append(description)

        #first image
        observed = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
        observed = cv2.resize(observed, (700, 280), interpolation=cv2.INTER_AREA)
        observed = ImageTk.PhotoImage(image=Image.fromarray(observed))
        label = tk.Label(frame_buttons, image=observed)
        label.image = observed
        label.grid(row=i, column=1, sticky='news', padx=xpadding, pady=ypadding)

        #second image
        expected = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
        expected = cv2.resize(expected, (700, 280), interpolation=cv2.INTER_AREA)
        expected = ImageTk.PhotoImage(image=Image.fromarray(expected))
        label = tk.Label(frame_buttons, image=expected)
        label.image = expected
        label.grid(row=i, column=2, sticky='news', padx=xpadding, pady=ypadding)

        """
        description = tk.Message(frame_buttons, text="Description of the images here")
        description.grid(row=i, column=0, padx=5, pady=5)
        messages.append(description)
        for j in range(1, columns):
            # buttons[i][j] = tk.Button(frame_buttons, text=("%d,%d" % (i + 1, j + 1)))
            # buttons[i][j].grid(row=i, column=j, sticky='news')
            # buttons[i][j] = tk.Label(frame_buttons, text="gridlabel", fg="green")
            # read in image
            observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png")
            observed = cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)
            observed = cv2.resize(observed, (700, 280), interpolation=cv2.INTER_AREA)
            observed = ImageTk.PhotoImage(image=Image.fromarray(observed))
            buttons[i][j] = tk.Label(frame_buttons, image=observed)
            buttons[i][j].image = observed
            buttons[i][j].grid(row=i, column=j, sticky='news', padx=xpadding, pady=ypadding)
        """
    # Update buttons frames idle tasks to let tkinter calculate buttons sizes
    frame_buttons.update_idletasks()

    # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
    first5columns_width = sum([buttons[0][j].winfo_width() + (xpadding * 2) for j in range(0, columns)]) + 100

    first5rows_height = sum([buttons[i][0].winfo_height() + (ypadding * 2) for i in range(0, columns)])
    first5rows_height = 900
    frame_canvas.config(width=first5columns_width + vsb.winfo_width(), height=first5rows_height)
    frame_canvas.config(width=1600, height=first5rows_height)

    # Set the canvas scrolling region
    canvas.config(scrollregion=canvas.bbox("all"))

    # Launch the GUI
    root.mainloop()


def build_gui(data):
    # setting up the tkinter gui
    root = tk.Tk()
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frame_main = tk.Frame(root, bg="gray")
    frame_main.grid(sticky='news')

    # column headers at the top
    label3 = tk.Label(frame_main, text="Info", fg="blue", bg='white')
    label3.grid(row=0, column=0, pady=5, sticky='')
    label1 = tk.Label(frame_main, text="Observed", fg="blue", bg='white')
    label1.grid(row=0, column=1, pady=(5, 0), sticky='')
    label2 = tk.Label(frame_main, text="Expected", fg="blue", bg='white')
    label2.grid(row=0, column=2, pady=(5, 0), sticky='')

    # Create a frame for the canvas with non-zero row&column weights
    frame_canvas = tk.Frame(frame_main)
    frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw', columnspan=3)
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    # Set grid_propagate to False to allow 5-by-5 buttons resizing later
    frame_canvas.grid_propagate(False)

    # Add a canvas in that frame
    canvas = tk.Canvas(frame_canvas, bg="gray")
    canvas.grid(row=0, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=vsb.set)

    # Create a frame to contain the buttons
    frame_buttons = tk.Frame(canvas, bg="gray")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')

    # setup frame
    rows = len(data)
    columns = 3
    xpadding = 10
    ypadding = 10
    buttons = [[tk.Label() for j in range(columns)] for i in range(len(data))]
    messages = []
    the_images_for_some_reason = []

    for i in range(0, rows):
        #get the data for this transformation
        desc, i1, i2 = data[i]

        #create description label
        description = tk.Message(frame_buttons, text=desc, width=100)
        description.grid(row=i, column=0, padx=5, pady=5)
        messages.append(description)

        #first image
        observed = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
        observed = cv2.resize(observed, (700, 280), interpolation=cv2.INTER_AREA)
        observed = ImageTk.PhotoImage(image=Image.fromarray(observed))
        label = tk.Label(frame_buttons, image=observed)
        label.image = observed
        label.grid(row=i, column=1, sticky='news', padx=xpadding, pady=ypadding)

        #second image
        expected = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
        expected = cv2.resize(expected, (700, 280), interpolation=cv2.INTER_AREA)
        expected = ImageTk.PhotoImage(image=Image.fromarray(expected))
        label = tk.Label(frame_buttons, image=expected)
        label.image = expected
        label.grid(row=i, column=2, sticky='news', padx=xpadding, pady=ypadding)

        """
        description = tk.Message(frame_buttons, text="Description of the images here")
        description.grid(row=i, column=0, padx=5, pady=5)
        messages.append(description)
        for j in range(1, columns):
            # buttons[i][j] = tk.Button(frame_buttons, text=("%d,%d" % (i + 1, j + 1)))
            # buttons[i][j].grid(row=i, column=j, sticky='news')
            # buttons[i][j] = tk.Label(frame_buttons, text="gridlabel", fg="green")
            # read in image
            observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png")
            observed = cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)
            observed = cv2.resize(observed, (700, 280), interpolation=cv2.INTER_AREA)
            observed = ImageTk.PhotoImage(image=Image.fromarray(observed))
            buttons[i][j] = tk.Label(frame_buttons, image=observed)
            buttons[i][j].image = observed
            buttons[i][j].grid(row=i, column=j, sticky='news', padx=xpadding, pady=ypadding)
        """
    # Update buttons frames idle tasks to let tkinter calculate buttons sizes
    frame_buttons.update_idletasks()

    # Resize the canvas frame to show the first 5 x 5 of cells
    #first5columns_width = sum([buttons[0][j].winfo_width() + (xpadding * 2) for j in range(0, columns)]) + 100
    #first5rows_height = sum([buttons[i][0].winfo_height() + (ypadding * 2) for i in range(0, columns)])
    #frame_canvas.config(width=first5columns_width + vsb.winfo_width(), height=first5rows_height)
    frame_canvas.config(width=1600, height=900)

    # Set the canvas scrolling region
    canvas.config(scrollregion=canvas.bbox("all"))

    # Launch the GUI
    root.mainloop()


def test10():
    # setup some test data
    images = [
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail1.jpg', cv2.IMREAD_GRAYSCALE),  # 0
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail2.jpg'),  # 1
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg'),  # 2
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail4.jpg'),  # 3
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail6.jpg'),  # 4
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png'),  # 5
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/dash.png'),  # 6
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/dash4.jpg'),  # 7
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/rawdisplay.png'),  # 8
        cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/SlicTest.png'),  # 9
    ]
    data = []
    for i in range(0, 9, 2):
        data.append((f'this is images {i + 1} and {i + 2}', images[i], images[i + 1]))


    #call the function
    build_gui(data)



def chi_function(a, b):
    a = a.flatten()
    b = b.flatten()

    sum = 0
    for i in range(len(a)):
        an, bn = int(a[i]), int(b[i])
        numerator = (an-bn) ** 2
        denominator = (an + bn + 1)
        sum += (numerator / denominator)
    return sum / 2

#COMPARES PICTURE AND SCREENSHOT
def test11():

    #grab the images I want to work with
    #observed = cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png')
    observed = cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/picture1.png')
    expected = cv2.imread('/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/screenshot1.png')

    #build place to store info to be displayed and place the initial ones on it
    data = [('original', observed, expected)]


    #median blur first
    medianblur = BlurFilter(expected, observed, BlurFilter.MEDIAN, 5)
    data.append(('5x5 Median Blur', medianblur.o_filtered, medianblur.e_filtered))

    #grab texture
    """
    o_texture = TextureFilter.texture_features(observed)  #grab multi dimensionsal texture
    e_texture = TextureFilter.texture_features(expected)
    o_texture = np.mean(o_texture, axis=2).astype(np.uint8) # mean composite
    e_texture = np.mean(e_texture, axis=2).astype(np.uint8)
    data.append(('Texture Composite', o_texture, e_texture))
    """

    #split into a grid
    rows, cols = 4, 8
    o_grid = ImageGrid(observed, rows, cols)
    e_grid = ImageGrid(expected, rows, cols)
    data.append(('partitioned into  8x4 grid', o_grid.mosaic(), e_grid.mosaic()))

    """
    #test placing some random image in there
    test_image = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/dash.png")
    o_grid.set(test_image, 1, 2)
    e_grid.set(test_image, 3, 4)
    data.append(('replaced top left corner for testing', o_grid.mosaic(), e_grid.mosaic()))
    """
    o_downscale = [[None for _ in range(cols)] for _ in range(rows)]
    e_downscale = [[None for _ in range(cols)] for _ in range(rows)]

    #Downsample the different places in the image grid
    # going down is done with area, going up is done with nearest exact
    for row in range(rows):
        for col in range(cols):
            o_subimage = o_grid.get(row, col)#grab the subimage to be modified
            e_subimage = e_grid.get(row, col)

            o_down = cv2.resize(o_subimage, (10, 10), interpolation=cv2.INTER_AREA)#downsample
            e_down = cv2.resize(e_subimage, (10, 10), interpolation=cv2.INTER_AREA)

            o_downscale[row][col] = o_down
            e_downscale[row][col] = e_down

            d = o_grid.get_subimage_shape() # dimensions to reshape back to

            o_up = cv2.resize(o_down, (d[1], d[0]), interpolation=cv2.INTER_NEAREST_EXACT)#back to the right size
            e_up = cv2.resize(e_down, (d[1], d[0]), interpolation=cv2.INTER_NEAREST_EXACT)

            o_grid.set(o_up, row, col)#set the subimage to the newly modified one
            e_grid.set(e_up, row, col)

    o_down_grid = ImageGrid(o_downscale, rows, cols)
    e_down_grid = ImageGrid(e_downscale, rows, cols)

    print("shape:", o_down_grid.get(0,0).shape)
    print("shape:", e_down_grid.get(0, 0).shape)

    data.append(('downsampled the subimages to 10 x 10, area interpolation', o_grid.mosaic(), e_grid.mosaic()))



    #now I can grab the chi squared for each one
    original_for_visualizing_stats = ImageGrid(np.copy(observed), rows, cols)

    chi_squared = []
    print('------------')
    for row in range(rows):
        row_data = []
        for col in range(cols):

            o_subimage = o_down_grid.get(row, col)
            e_subimage = e_down_grid.get(row, col)


            """
            #chi = chisquare(o_subimage, e_subimage)
            chi = 0.5 * np.sum([((o_subimage - e_subimage) ** 2) / (o_subimage + e_subimage) for (o_subimage, e_subimage) in zip(o_subimage, e_subimage)])
            row_data.append(chi)
            """


            """
            diff = np.subtract(o_subimage, e_subimage)
            numerator = diff * diff
            denominator = o_subimage + e_subimage + 1
            if np.isin(0, denominator):
                print('zero here:', denominator)
            s = np.sum(numerator/denominator)
            row_data.append(s)
            """
            #my own function
            s = chi_function(o_subimage, e_subimage)
            row_data.append(s)
        chi_squared.append(row_data)

    #Draw onto the grid
    for row in range(rows):
        for col in range(cols):
            subimage = original_for_visualizing_stats.get(row, col)
            subimage = cv2.putText(subimage, str(int(chi_squared[row][col])), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                   (0, 0, 255), 3, cv2.LINE_AA)
            original_for_visualizing_stats.set(subimage, row, col)

    data.append(('Chi-Squared results', observed, original_for_visualizing_stats.mosaic()))

    # build the visualization
    build_gui(data)



















if __name__ == '__main__':
    """run tests"""
    test11()

#next week, results done on framework
#based on those results, continue refining, new combination of processing steps
#Whitepaper
#explanation of what happens when stuff occurs, why is that the best method.
#Can I tweak those to make it even better.
#make sure to break it down

#measure how active a region is, edge detector.
#see if edges have disappeared
#check book for the different edge detectors


#for next week:
# More sample pictures with the screens in the lab
# more variety of tests done and recorded (organized)
# begin incorporating some for of edge detector, how 'active' a region is


#some notes:
# saliency -> like how 'active' the region is
# add particular noise pattern to image that can be picked up by the camera
# insert black frame without the viewer noticing