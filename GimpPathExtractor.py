#!/Applications/GIMP-2.10.app/Contents/MacOS/python
from gimpfu import *
import gtk


"""
# ----original code----
image_number = 0
img = gimp.image_list()[image_number]
path = img.vectors[0]
stroke_ids = pdb.gimp_vectors_get_strokes(path)
_, _, coords, _ = pdb.gimp_vectors_stroke_get_points(path, stroke_ids[1][0])
for i in range(0, len(coords), 6):
    x, y = coords[i], coords[i + 1]
    print("[" + str(x) + ", " + str(y) + "],")
"""



def export_path_points(image, drawable):
    # Get the first path (vectors)
    if not image.vectors:
        message = "No path found in this image!"
    else:
        path = image.vectors[0]
        stroke_ids = pdb.gimp_vectors_get_strokes(path)
        _, _, coords, _ = pdb.gimp_vectors_stroke_get_points(path, stroke_ids[1][0])
        message = ""
        for i in range(0, len(coords), 6):
            x, y = coords[i], coords[i + 1]
            message += "[{}, {}],\n".format(x, y)

    # Display the output in a GTK dialog window
    window = gtk.Window()
    window.set_title("Path Points")
    window.set_default_size(300, 200)
    window.connect('destroy', lambda _: gtk.main_quit())
    textview = gtk.TextView()
    textbuffer = textview.get_buffer()
    textbuffer.set_text(message)
    textview.set_editable(False)
    scroll = gtk.ScrolledWindow()
    scroll.add(textview)
    window.add(scroll)
    window.show_all()
    gtk.main()

register(
    "python_fu_export_path_points",
    "Export Path Points",
    "Show coordinates of points in the first path in a GTK window.",
    "Aidan Lear", "Your Name", "2024",
    "Export Path Points...",
    "*",
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    export_path_points,
    menu="<Image>/Filters"
)

main()

