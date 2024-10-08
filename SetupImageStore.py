import cv2
import os



def first_thing():


    background_folder = "/Users/aidan/Desktop/Backgrounds/"


    #list files in mypath
    mypath = "/Users/aidan/Desktop/PicsFromPhone/NewImageSet/"
    observed = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        observed.extend(filenames)
    observed = [os.path.join(mypath, p) for p in observed]






    #Expected and obstruction masks
    generated_set = []
    gen_path = "/Users/aidan/Desktop/GeneratedSet"
    for (dirpath, dirnames, filenames) in os.walk(gen_path):
        generated_set.extend(filenames)
    generated_set = [os.path.join(gen_path, p) for p in generated_set]

    #Sort into masks or not
    not_masks = []
    masks = []
    for path in generated_set:
        if 'mask' in path:
            masks.append(path)
        else:
            not_masks.append(path)
    generated_set = not_masks[:]




    from AlignImages import align_images


    #iterate over the generated set images, grab things that relate to it.
    test_id = 0
    for path in generated_set:
        test_id += 1
        #skip over any extraneous files, e.g. .DS_Store
        if not (path.endswith(".png")):
            continue

        print("----"*20)
        print("Generated Image:", path)
        #grab corresponding picture path
        the_three_numbers = [s for s in path if s.isdigit()]


        is_obstructed = len(the_three_numbers) == 3
        print("Obstructed:", is_obstructed)


        observed_path = os.path.join("/Users/aidan/Desktop/PicsFromPhone/NewImageSetCropped/", "".join(the_three_numbers) + ".jpg")
        if not is_obstructed:
            observed_path = os.path.join("/Users/aidan/Desktop/PicsFromPhone/NewImageSetCropped/", "".join(the_three_numbers) + "0.jpg")
        print("Observed:", observed_path)


        dashboard_number = the_three_numbers[0]
        print("Dashboard Number:", dashboard_number)


        dashboard_path = background_folder + str(dashboard_number) + ".png"
        print("Dashboard/Expected:", dashboard_path)


        if is_obstructed:
            a, b, c = the_three_numbers
            obstruction_path = f"/Users/aidan/Desktop/GeneratedSet/mask-dash{a}-obstruction{b}-{c}.png"
            print("Obstruction Mask Path:", obstruction_path)
        else:
            a, b = the_three_numbers
            obstruction_path = f"/Users/aidan/Desktop/GeneratedSet/mask-dash{a}-obstruction0.png"
            print("Obstruction Mask Path:", obstruction_path)

        # Align and save the image
        o, e = cv2.imread(observed_path), cv2.imread(dashboard_path)
        observed_aligned = align_images(o, e)
        cv2.imwrite(f"/Users/aidan/Desktop/ExpoSet/observed/{test_id}.png", observed_aligned)
        cv2.imwrite(f"/Users/aidan/Desktop/ExpoSet/expected/{test_id}.png", e)
        cv2.imwrite(f"/Users/aidan/Desktop/ExpoSet/mask/{test_id}.png", cv2.imread(obstruction_path))


#first_thing()











