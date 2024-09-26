def copy_file(json_path, destination_folder):
    image_path = json_path.replace(".json", ".jpg")
    
    if not os.path.exists(image_path):
        if not os.path.exists(image_path.replace(".jpg", ".jpeg")):
            print(f"Image {image_path} does not exist")
            return
        else:
            image_path = image_path.replace(".jpg", ".jpeg")

    
    if not os.path.exists(json_path):
        print(f"JSON {json_path} does not exist")
        return
    

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    if os.path.exists(os.path.join(destination_folder, os.path.basename(json_path))):
        print(f"File {json_path} already exists in {destination_folder}")
        return
    
    if os.path.exists(os.path.join(destination_folder, os.path.basename(image_path))):
        print(f"File {image_path} already exists in {destination_folder}")
        return

    shutil.copy(json_path, destination_folder)
    shutil.copy(image_path, destination_folder)

    #delete EA files
    json_dest = os.path.join(destination_folder, os.path.basename(json_path))
    image_dest = os.path.join(destination_folder, os.path.basename(image_path))

    ea_path_json = json_dest.split("/")[:-1] + ["._" + json_dest.split("/")[-1]]
    ea_path_json = "/".join(ea_path_json)
    
    ea_path_image = image_dest.split("/")[:-1] + ["._" + image_dest.split("/")[-1]]
    ea_path_image = "/".join(ea_path_image)

    if os.path.exists(ea_path_json):
        os.remove(ea_path_json)
    
    if os.path.exists(ea_path_image):
        os.remove(ea_path_image)

def copy_single_batch(all_files, destination_folder):
    for json_path in tqdm(all_files, desc=f"Copying files to {destination_folder}"):
        copy_file(json_path, destination_folder)

def copy_batches(all_files, destination_folder, batch_size):
    def process_batch(batch_files, batch_folder):
        copy_single_batch(batch_files, batch_folder)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i+batch_size]
            batch_folder = os.path.join(destination_folder, f"Batch_{i//batch_size}")
            futures.append(executor.submit(process_batch, batch_files, batch_folder))

        for future in as_completed(futures):
            future.result()  # To raise any exceptions that occurred during copying



def copy(source_paths, dst,batch_size):
    if not os.path.exists(dst):
        print(f"Desination folder {dst} does not exist")
        return

    last_folder = source_paths[0].split("/")[-2]

    if not os.path.exists(os.path.join(dst, last_folder)):
        os.makedirs(os.path.join(dst, last_folder))
        print(f"Created folder {os.path.join(dst, last_folder)}")
    else:
        print(f"Folder {os.path.join(dst, last_folder)} already exists")
        return
    
    copy_batches(source_paths, dst, batch_size)
    


    